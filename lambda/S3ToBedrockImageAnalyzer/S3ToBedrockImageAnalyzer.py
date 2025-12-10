import json
import boto3
import urllib.parse
import base64
import os
import logging
from datetime import datetime
import time
import io
from botocore.exceptions import ClientError
from botocore.config import Config

# ロギング設定
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 環境変数から設定を取得
# Claude 3 Haiku のモデルID
ANALYSIS_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID') 
# CDKで設定したS3バケット名
GENERATED_IMAGE_BUCKET_NAME = os.environ.get('GENERATED_IMAGE_BUCKET_NAME')
# 画像生成モデルID (固定値として利用)
GENERATION_MODEL_ID = 'amazon.nova-canvas-v1:0'

# AWSクライアントの初期化
s3_client = boto3.client('s3')
# Bedrock runtime client with extended timeout for image generation
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    config=Config(read_timeout=300)
)

def handler(event, context):
    """
    S3へのファイルアップロードをトリガーとして実行されるLambdaハンドラ関数。
    画像を分析し、そのテキスト結果を使って間取り図の画像を生成します。
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
        
        logger.info(f"Image read. MIME Type: {mime_type}. Starting Analysis (Step 1/2)...")

        # 2. Step 1: Claudeで画像分析を実行し、テキストプロンプトを生成
        analysis_prompt_text = (
            "You are an expert architect. Analyze this photo of a room. "
            "Describe the layout, dimensions, and contents in extreme detail suitable for generating a floor plan. "
            "Output *only* the single, continuous text prompt for an image generator to create a simple, minimalist 2D floor plan based on this room's geometry and contents."
        )
        
        generated_prompt = invoke_bedrock_multimodal_analysis(image_base64, mime_type, analysis_prompt_text)
        
        logger.info(f"Analysis complete. Generated Prompt (Step 2/2): {generated_prompt[:200]}...")

        # 3. Step 2: Stable Diffusion XLで間取り図の画像を生成
        image_base64 = invoke_bedrock_sdxl_generation(generated_prompt)

        logger.info(f"Image generation complete. Saving to S3...")
        
        # 4. 生成された画像をS3に保存
        image_binary = base64.b64decode(image_base64)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_key = f"generated_floorplan_from_{key.replace('/', '_')}_{timestamp}.png"
        
        s3_client.put_object(
            Bucket=GENERATED_IMAGE_BUCKET_NAME,
            Key=output_key,
            Body=image_binary,
            ContentType='image/png'
        )
        
        logger.info(f"--- Image Generation Complete ---")
        logger.info(f"Generated floor plan saved to s3://{GENERATED_IMAGE_BUCKET_NAME}/{output_key}")
            
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Floor plan generation successful', 'output_key': output_key})
        }

    except Exception as e:
        logger.error(f"Error processing S3 event or invoking Bedrock: {e}")
        raise e

# --- ヘルパー関数: S3画像読み込み (再掲) ---

def get_image_from_s3(bucket: str, key: str):
    """S3から画像ファイルを読み込み、MIMEタイプを推定する"""
    
    # MIMEタイプを判別するためのファイル拡張子チェック (S3クライアント呼び出し前に実行)
    if key.lower().endswith(('.jpg', '.jpeg')):
        expected_mime_type = 'image/jpeg'
    elif key.lower().endswith('.png'):
        expected_mime_type = 'image/png'
    else:
        logger.error(f"Unsupported file type: {key}. Only .jpg and .png are supported.")
        # サポートされていないファイルの場合は、ここでエラーを発生させる
        raise ValueError(f"Unsupported file type for analysis: {key}")

    try:
        # S3からオブジェクトを取得
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
        
        # S3から取得したContentTypeをそのまま使用（または推定値を使用）
        mime_type = response.get('ContentType', expected_mime_type)
        
        return image_bytes, mime_type

    except ClientError as e:
        # S3のクライアントエラー（ファイルアクセス権限やファイルなし）を補足
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.error(f"S3 Object Not Found: s3://{bucket}/{key}")
        elif error_code == 'AccessDenied':
            logger.error(f"S3 Access Denied: Check Lambda IAM role for s3:GetObject on {bucket}")
        else:
            logger.error(f"Unknown S3 ClientError: {e}")
        # S3の読み込みに失敗した場合、呼び出し元にエラーを再送出する
        raise 
    except Exception as e:
        logger.error(f"Unexpected error in get_image_from_s3: {e}")
        raise

# --- ヘルパー関数: Claudeによる分析 (修正) ---

def invoke_bedrock_multimodal_analysis(image_base64: str, mime_type: str, prompt_text: str) -> str:
    """Bedrockのマルチモーダルモデル (Haiku) を呼び出して分析を実行する"""
    
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
        "max_tokens": 2048, # プロンプト生成に十分なトークン数
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
            return "minimalist 2D floor plan, white background, black lines, architectural drawing"

    except Exception as e:
        logger.error(f"Bedrock Claude API invocation failed: {e}")
        raise e

# --- ヘルパー関数: Nova Canvas による画像生成 ---

def invoke_bedrock_sdxl_generation(prompt: str) -> str:
    """Nova Canvas モデルを呼び出して画像を生成する"""
    
    # プロンプトに間取り図のスタイルを追加して強調
    full_prompt = f"minimalist 2D architectural floor plan, black lines on white, scale accurate, detailed, {prompt}"
    
    # Nova Canvas API (InvokeModel) のペイロード
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": full_prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 720,
            "width": 1280,
            "cfgScale": 7.0,
            "seed": int(time.time())
        }
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=GENERATION_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        if response_body.get('images') and len(response_body['images']) > 0:
            return response_body['images'][0]
        else:
            logger.warning(f"Nova Canvas generation response structure unexpected: {response_body}")
            raise RuntimeError("Image generation failed: No image data returned.")

    except Exception as e:
        logger.error(f"Bedrock Nova Canvas API invocation failed: {e}")
        raise e