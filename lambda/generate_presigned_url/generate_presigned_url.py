import json
import boto3
import os
from datetime import datetime
from botocore.config import Config
import uuid

# S3クライアントを初期化
# https://roskothecat.com/en/posts/s3-presigned-cors-error/
s3_client = boto3.client(
    "s3",
    config=Config(
        signature_version="s3v4",
        region_name=os.environ.get("AWS_REGION", "ap-northeast-1"),
        s3={"addressing_style": "virtual"},
    ),
)


def handler(event, context):
    """
    S3の署名付きURLを生成するLambda関数

    リクエストパラメータ:
        - file_extension: ファイルの拡張子 (例: jpg, png)

    レスポンス:
        - upload_url: アップロード用の署名付きURL
        - file_key: S3に保存されるファイルのキー（パス）
    """
    try:
        # 環境変数からS3バケット名を取得
        bucket_name = os.environ["BUCKET_NAME"]

        # リクエストボディを解析
        if "body" in event:
            body = (
                json.loads(event["body"])
                if isinstance(event["body"], str)
                else event["body"]
            )
        else:
            body = event

        # ファイル拡張子を取得（デフォルト: jpg）
        file_extension = body.get("file_extension", "jpg")

        # ユニークなファイル名を生成（タイムスタンプ + UUID）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_key = f"camera_images/{timestamp}_{unique_id}.{file_extension}"

        # 署名付きURLを生成（有効期限: 300秒 = 5分）
        # ContentTypeを指定しないことで、アップロード時のヘッダー送信を不要にし、
        # プリフライトリクエストを回避してCORSエラーを防ぐ
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": bucket_name,
                "Key": file_key,
            },
            ExpiresIn=3600,
        )

        print(presigned_url)

        # レスポンスを返す
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps(
                {
                    "upload_url": presigned_url,
                    "file_key": file_key,
                    "bucket_name": bucket_name,
                    "message": "署名付きURLの生成に成功しました",
                },
                ensure_ascii=False,
            ),
        }

    except Exception as e:
        # エラー時のレスポンス
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps(
                {"error": str(e), "message": "署名付きURLの生成に失敗しました"},
                ensure_ascii=False,
            ),
        }
