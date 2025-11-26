# JSON形式のデータを扱うための標準ライブラリ
import json
# AWSサービスを操作するためのPython SDK
import boto3

# Amazon Bedrock(生成AI)を操作するクライアントを作成
# region_name: 使用するAWSリージョン(米国西部)
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")

# 使用するAIモデルのID(Claude 3.5 Haiku)
# 変更する場合は、Bedrockのコンソールで確認して更新する
# https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/inference-profiles
modelId = "us.anthropic.claude-3-5-haiku-20241022-v1:0"


def handler(event, context):
    """
    Lambda関数のメイン処理

    【機能】
    日記のタイトルを受け取り、Amazon BedrockのAIで日記の内容を自動生成して返す

    【引数】
    event: Lambda関数に渡されるイベントデータ(HTTPリクエスト情報を含む)
    context: Lambda実行環境の情報(この関数では未使用)

    【戻り値】
    HTTPレスポンス(ステータスコード、ヘッダー、ボディを含む辞書)
    """
    try:
        # ========================================
        # ステップ1: リクエストからタイトルを取り出す
        # ========================================
        # event["body"]にはJSON文字列が入っているので、Pythonの辞書に変換
        body = json.loads(event.get("body", "{}"))
        # 辞書から"title"キーの値を取得(なければ空文字列)
        title = body.get("title", "")

        # タイトルが空の場合はエラーレスポンスを返す
        if not title:
            return {
                "statusCode": 400,  # HTTPステータスコード: 400 Bad Request
                "headers": {
                    "Content-Type": "application/json",  # JSON形式で返す
                },
                "body": json.dumps({"error": "タイトルが必要です"}),
            }

        # ========================================
        # ステップ2: AIに送るプロンプト(指示文)を作成
        # ========================================
        # f文字列を使ってタイトルを埋め込んだプロンプトを作成
        prompt = f"""以下の日記のタイトルに基づいて、適切な日記の内容を日本語で生成してください。自然で心のこもった内容にしてください。タイトル: {title}日記の内容:"""

        # BedrockのAPIに送るメッセージ形式を作成
        # role: "user"はユーザーからのメッセージを意味する
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        # ========================================
        # ステップ3: Amazon BedrockのAIを呼び出す
        # ========================================
        # converse()メソッドでAIモデルと会話
        response = bedrock_runtime.converse(modelId=modelId, messages=messages)

        # ========================================
        # ステップ4: AIの返答から生成された日記内容を取り出す
        # ========================================
        # レスポンスのネストした構造から、テキスト部分だけを抽出
        generated_content = response["output"]["message"]["content"][0]["text"]

        # ========================================
        # ステップ5: 成功レスポンスを返す
        # ========================================
        return {
            "statusCode": 200,  # HTTPステータスコード: 200 OK(成功)
            "headers": {
                "Content-Type": "application/json",  # JSON形式で返す
            },
            # タイトルと生成された内容をJSON形式で返す
            # ensure_ascii=False: 日本語をそのまま出力(エスケープしない)
            "body": json.dumps(
                {"title": title, "content": generated_content}, ensure_ascii=False
            ),
        }

    except Exception as e:
        # ========================================
        # エラーハンドリング: 何か問題が発生した場合
        # ========================================
        # エラー内容をログに出力(CloudWatch Logsで確認可能)
        print(f"Error: {str(e)}")

        # エラーレスポンスを返す
        return {
            "statusCode": 500,  # HTTPステータスコード: 500 Internal Server Error
            "headers": {
                "Content-Type": "application/json",
            },
            # エラーメッセージをJSON形式で返す
            "body": json.dumps({"error": str(e)}, ensure_ascii=False),
        }
