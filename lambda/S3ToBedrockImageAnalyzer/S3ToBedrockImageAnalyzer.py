import json
import boto3
import urllib.parse
import base64
import os
import logging
from datetime import datetime
import time
from botocore.exceptions import ClientError
from botocore.config import Config

# ロギング設定
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 環境変数から設定を取得
# Claude Sonnet 4.5 のモデルID
ANALYSIS_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID') 
# CDKで設定したS3バケット名
GENERATED_IMAGE_BUCKET_NAME = os.environ.get('GENERATED_IMAGE_BUCKET_NAME')

# AWSクライアントの初期化
s3_client = boto3.client('s3')
# Bedrock runtime client with extended timeout for image generation
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    config=Config(read_timeout=300)
)

# --- 間取り図のCSS定義 ---
# ClaudeはこのCSSを元に、<body>内の構造をゼロから生成します。
CSS_CONTENT = """
<style>
  /* 間取り図全体のコンテナ */
  .madori-container {
    width: 100%;
    max-width: 600px;
    margin: 20px auto;
    border: 4px solid #333;
    background-color: #fff;
    font-family: "Hiragino Kaku Gothic ProN", "Meiryo", sans-serif;
    
    /* グリッドレイアウトの設定 (この値を分析結果に合わせて変更すること) */
    display: grid;
    gap: 2px;
    background-color: #333;
  }

  /* 部屋ごとの共通スタイル */
  .room {
    background-color: #fff;
    display: flex;
    flex-direction: column;
    align-content: center;
    justify-content: center;
    text-align: center;
    padding: 10px;
    position: relative;
    box-sizing: border-box; 
  }

  /* 部屋ラベルの文字サイズ */
  .label {
    font-weight: bold;
    font-size: 1.1em;
    margin-bottom: 5px;
  }
  
  .tatami {
    font-size: 0.8em;
    color: #666;
  }

  /* --- 個別の部屋エリア指定 (a.htmlの内容を基に設定) --- */
  /* Claudeはgrid-areaをこれらから選択し、grid-template-areasを動的に定義します */
  .area-room1 { grid-area: room1; background-color: #f9f9f9; }
  .area-room2 { grid-area: room2; background-color: #f9f9f9; }
  .area-room3 { grid-area: room3; background-color: #e8f5e9; }
  .area-ldk { grid-area: ldk; background-color: #fff8e1; border-left: 1px dashed #ccc; }
  .area-bath { grid-area: bath; background-color: #e3f2fd; }
  .area-entrance { grid-area: entrance; background-color: #ddd; }
  .area-balcony { 
      grid-area: balcony; 
      background-color: #eee;
      height: 60px; 
      border-top: 2px double #333; 
  }

  /* ドアの表現 */
  .door-mark::after {
    content: "";
    display: block;
    width: 30px;
    height: 4px;
    background-color: #8d6e63;
    margin: 5px auto;
    border-radius: 2px;
  }
</style>
"""

# --- Claude 3 Haiku への完全生成指示プロンプト ---
ANALYSIS_PROMPT_INSTRUCTION = f"""
あなたは専門の建築家であり、HTML/CSSの専門家です。
提供された部屋の画像を分析し、そのレイアウト、部屋数、推定される寸法（帖数を含む）に基づいて、**間取り図の構造をゼロから記述した完全なHTMLコードを生成してください。**

【厳守事項】
1.  出力は、**<!DOCTYPE html>から</html>まで** の完全なHTMLコード**のみ**としてください。余分な説明やMarkdown（```html）は一切禁止です。
2.  生成するHTMLの<head>要素内に、以下の【必須CSS】をそのままコピー＆ペーストして組み込んでください。
3.  部屋の配置、大きさの比率、部屋の名称（LDK、洋室、Bath/WCなど）、および推定される帖数（〇.〇帖）を分析結果に基づいて正確に反映させてください。
4.  CSS Gridの 'grid-template-columns', 'grid-template-rows', 'grid-template-areas' の値を、**分析した間取りに合わせて動的に**調整してください。

【必須CSS】
{CSS_CONTENT}
"""

def handler(event, context):
    """
    S3へのファイルアップロードをトリガーとして実行されるLambdaハンドラ関数。
    画像を分析し、そのテキスト結果（HTML）をS3に保存します。
    """
    if not ANALYSIS_MODEL_ID or not GENERATED_IMAGE_BUCKET_NAME:
        logger.error("Configuration Error: Required environment variables are missing.")
        raise ValueError("Configuration Error: BEDROCK_MODEL_ID or GENERATED_IMAGE_BUCKET_NAME missing.")
        
    logger.info(f"Received event: {event}")

    try:
        # 1. S3イベントからバケット名とオブジェクトキーを取得し、画像データを取得
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')
        
        logger.info(f"--- Processing Input Image: s3://{bucket}/{key} ---")

        # 画像ファイルの読み込み
        image_bytes, mime_type = get_image_from_s3(bucket, key)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        logger.info(f"Image read. MIME Type: {mime_type}. Starting Analysis (HTML Generation)...")

        # 2. Claudeで画像分析を実行し、HTML間取り図コードを生成
        
        # Claudeに画像を分析させ、その結果をHTMLとして生成させる
        html_content = invoke_bedrock_multimodal_analysis(image_base64, mime_type, ANALYSIS_PROMPT_INSTRUCTION)
        
        logger.info(f"HTML generation complete. Saving to S3...")
        
        # 3. 生成されたHTMLコードをS3に保存
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # 拡張子を .html に変更
        output_key = f"generated_floorplan_from_{key.replace('/', '_')}_{timestamp}.html" 
        
        # strをbytesに変換してS3に保存
        html_bytes = html_content.encode('utf-8') 
        
        s3_client.put_object(
            Bucket=GENERATED_IMAGE_BUCKET_NAME,
            Key=output_key,
            Body=html_bytes, # バイトデータを格納
            ContentType='text/html' # コンテンツタイプを text/html に変更
        )
        
        logger.info(f"--- HTML Floor Plan Generation Complete ---")
        logger.info(f"Generated HTML floor plan saved to s3://{GENERATED_IMAGE_BUCKET_NAME}/{output_key}")
            
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'HTML Floor plan generation successful', 'output_key': output_key})
        }

    except Exception as e:
        logger.error(f"Error processing S3 event or invoking Bedrock: {e}")
        raise e

# --- ヘルパー関数: S3画像読み込み (変更なし) ---

def get_image_from_s3(bucket: str, key: str):
    """S3から画像ファイルを読み込み、MIMEタイプを推定する"""
    
    if key.lower().endswith(('.jpg', '.jpeg')):
        expected_mime_type = 'image/jpeg'
    elif key.lower().endswith('.png'):
        expected_mime_type = 'image/png'
    else:
        logger.error(f"Unsupported file type: {key}. Only .jpg and .png are supported.")
        raise ValueError(f"Unsupported file type for analysis: {key}")

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
        mime_type = response.get('ContentType', expected_mime_type)
        return image_bytes, mime_type

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.error(f"S3 Object Not Found: s3://{bucket}/{key}")
        elif error_code == 'AccessDenied':
            logger.error(f"S3 Access Denied: Check Lambda IAM role for s3:GetObject on {bucket}")
        else:
            logger.error(f"Unknown S3 ClientError: {e}")
        raise 
    except Exception as e:
        logger.error(f"Unexpected error in get_image_from_s3: {e}")
        raise

# --- ヘルパー関数: Claudeによる分析 (HTML出力対応) ---

def invoke_bedrock_multimodal_analysis(image_base64: str, mime_type: str, prompt_text: str) -> str:
    """Bedrockのマルチモーダルモデル (Haiku) を呼び出して分析を実行し、HTMLコードを返す"""
    
    # Bedrock API (InvokeModel) のペイロードを作成 (Anthropic Claude形式)
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        "max_tokens": 4096, # HTMLコード生成のため、最大トークン数を増やす
        "anthropic_version": "bedrock-2023-05-31"
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=ANALYSIS_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        if response_body.get('content') and response_body['content'][0].get('text'):
            return response_body['content'][0]['text']
        else:
            logger.warning(f"Claude analysis response structure unexpected: {response_body}")
            return "<html><body><h1>Error: Could not generate floor plan HTML.</h1></body></html>"

    except Exception as e:
        logger.error(f"Bedrock Claude API invocation failed: {e}")
        raise e