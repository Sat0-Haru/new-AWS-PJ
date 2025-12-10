import json
import boto3
import urllib.parse
import base64
import os
import logging

# ロギング設定
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 環境変数から設定を取得
# Bedrockで画像解析に使用するモデルID（例: anthropic.claude-3-sonnet-20240229-v1:0）
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID')

# AWSクライアントの初期化
s3_client = boto3.client('s3')
# Bedrockランタイムクライアントは、Lambdaが動作しているリージョンで自動的に初期化されます
bedrock_runtime = boto3.client('bedrock-runtime')

def handler(event, context):
    """
    S3へのファイルアップロードをトリガーとして実行されるLambdaハンドラ関数。
    """
    if not BEDROCK_MODEL_ID:
        logger.error("BEDROCK_MODEL_ID is not set in environment variables.")
        raise ValueError("Configuration Error: BEDROCK_MODEL_ID missing.")
        
    logger.info(f"Received event: {event}")

    try:
        # S3イベントからバケット名とオブジェクトキーを取得
        # S3イベントには複数のレコードが含まれる場合があるため、最初のレコードを使用
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        # キーにスペースなどURLエンコードされた文字がある場合に対応
        key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')
        
        logger.info(f"--- Processing Image: s3://{bucket}/{key} ---")

        # 1. S3から画像ファイルを読み込み、Base64でエンコード
        image_bytes, mime_type = get_image_from_s3(bucket, key)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        logger.info(f"Image read and encoded. MIME Type: {mime_type}")

        # 2. Amazon Bedrockを呼び出して画像分析を実行
        layout_content = invoke_bedrock_multimodal(image_base64, mime_type)
        
        # 3. 結果のログ出力（今回はDynamoDB保存をスキップし、ログ出力で確認）
        logger.info("--- Bedrock Analysis Complete ---")
        logger.info(f"Generated Layout Content (Partial): {layout_content[:500]}...") # 長すぎる場合を考慮
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Bedrock image analysis successful', 'image_key': key})
        }

    except Exception as e:
        logger.error(f"Error processing S3 event or invoking Bedrock: {e}")
        # Lambda関数は失敗し、CloudWatch Logsにエラーが記録されます
        raise e

# --- ヘルパー関数 ---

def get_image_from_s3(bucket: str, key: str):
    """S3から画像ファイルを読み込み、MIMEタイプを推定する"""
    try:
        image_object = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = image_object['Body'].read()

        # キーのサフィックスからMIMEタイプを推定
        if key.lower().endswith(('.png')):
            mime_type = 'image/png'
        elif key.lower().endswith(('.jpeg', '.jpg')):
            mime_type = 'image/jpeg'
        else:
            # サポートされていない形式の場合はエラーを出すか、デフォルトを設定
            raise ValueError(f"Unsupported file type for key: {key}")
            
        return image_bytes, mime_type
    except Exception as e:
        logger.error(f"Failed to read image from S3: {e}")
        raise e

def invoke_bedrock_multimodal(image_base64: str, mime_type: str) -> str:
    """Bedrockのマルチモーダルモデルを呼び出す"""
    
    # 画像分析と間取り図生成のためのプロンプト
    prompt_text = (
        "この部屋の写真を詳細に分析し、ドア、窓の位置と数を検出し、部屋の広さを合理的に推測してください。"
        "そして、これらの情報から推定される間取り図を**Markdown形式のテキスト**または**シンプルなSVG形式**で出力してください。"
        "回答は、解析結果を格納したJSONオブジェクトとして返してください。JSONのキーは 'layout_plan' とし、値に間取り図の情報を格納してください。"
    )

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
        "max_tokens": 4096,
        "anthropic_version": "bedrock-2023-05-31"
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        # Bedrockのレスポンスを処理
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        # Claude 3/4.5のレスポンスからテキストコンテンツを抽出
        if response_body.get('content') and response_body['content'][0].get('text'):
            return response_body['content'][0]['text']
        else:
            logger.warning(f"Bedrock response structure unexpected: {response_body}")
            return "Analysis failed: Unexpected model response structure."

    except Exception as e:
        logger.error(f"Bedrock API invocation failed: {e}")
        raise e