import json
import boto3
import urllib.parse
import base64
import os

# 環境変数または直接設定
AWS_REGION = os.environ.get('AWS_REGION', 'ap-northeast-1') # お使いの環境に合わせて変更してください
BEDROCK_MODEL_ID = 'anthropic.claude-sonnet-4-5-20250929-v1:0'

s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

def lambda_handler(event, context):
    try:
        # S3イベントからバケット名とオブジェクトキーを取得
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        # キーにスペースなどURLエンコードされた文字がある場合に対応
        key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')
        
        print(f"Processing image: s3://{bucket}/{key}")

        # 1. S3から画像ファイルを読み込み、Base64でエンコード
        image_object = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = image_object['Body'].read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # MIMEタイプの推定
        if key.lower().endswith(('.png')):
            mime_type = 'image/png'
        else:
            mime_type = 'image/jpeg'
            
        # 2. Bedrockへ渡すプロンプトと設定の準備 (Claude 3 Visionの例)
        prompt_text = (
            "この部屋の写真を詳細に分析し、ドア、窓の位置と数を検出し、部屋の広さを合理的に推測してください。"+
            "そして、これらの情報から推定される間取り図のテキストまたはシンプルなSVG形式で出力してください。"+
            "出力はJSON形式でお願いします。JSONのキーは 'layout_plan' とし、値に間取り図の情報を格納してください。"
        )
        
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

        # 3. Amazon BedrockのAPIを呼び出す
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        # 4. Bedrockのレスポンスを処理
        response_body = json.loads(response['body'].read().decode('utf-8'))
        layout_content = response_body['content'][0]['text']
        
        print("Bedrock output:", layout_content)
        
        # TODO: ステップ3として、この 'layout_content' をS3などに保存する処理を追加します。
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Bedrock processing complete', 'layout_plan': layout_content})
        }

    except Exception as e:
        print(f"Error processing S3 event or invoking Bedrock: {e}")
        # エラーログを残し、関数を終了
        raise e
