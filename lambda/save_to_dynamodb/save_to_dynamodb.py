# JSON形式のデータを扱うための標準ライブラリ
import json
# AWSサービスを操作するためのPython SDK
import boto3
# 日時を扱うための標準ライブラリ
from datetime import datetime
# 環境変数を扱うための標準ライブラリ
import os

# DynamoDBを操作するためのリソースオブジェクトを作成
dynamodb = boto3.resource('dynamodb')

# 環境変数からDynamoDBテーブル名を取得
# この環境変数はCDKスタックで設定されている(lib/aws_project_2025_template-stack.ts:85)
table_name = os.environ['TABLE_NAME']

# 指定されたテーブルに接続
table = dynamodb.Table(table_name)

def handler(event, context):
    """
    Lambda関数のメイン処理

    【機能】
    日記データ(日付、タイトル、内容)を受け取り、DynamoDBに保存する

    【引数】
    event: Lambda関数に渡されるイベントデータ(HTTPリクエスト情報を含む)
    context: Lambda実行環境の情報(この関数では未使用)

    【戻り値】
    HTTPレスポンス(ステータスコード、ヘッダー、ボディを含む辞書)
    """
    # 受信したイベント内容をログに出力(CloudWatch Logsで確認可能)
    print(f"Received event: {json.dumps(event)}")

    try:
        # ========================================
        # ステップ1: リクエストからデータを取り出す
        # ========================================
        # event['body']にはJSON文字列が入っているので、Pythonの辞書に変換
        # bodyが存在しない場合は空の辞書を使用
        body = json.loads(event.get('body', '{}')) if event.get('body') else {}

        # リクエストボディから日付、タイトル、内容を取得
        date = body.get('date')        # 日記の日付(例: "2025-01-15")
        title = body.get('title')      # 日記のタイトル
        content = body.get('content')  # 日記の内容

        # ========================================
        # ステップ2: 入力値のバリデーション(検証)
        # ========================================
        # date, title, contentがすべて存在するかチェック
        # all()関数: リスト内のすべての要素がTrueの場合にTrueを返す
        if not all([date, title, content]):
            # いずれかが欠けている場合はエラーレスポンスを返す
            return {
                'statusCode': 400,  # HTTPステータスコード: 400 Bad Request
                'headers': {
                    'Content-Type': 'application/json',
                },
                'body': json.dumps({
                    'error': 'date, title, and content are required'
                })
            }

        # ========================================
        # ステップ3: DynamoDBに保存するデータを作成
        # ========================================
        # 現在時刻を取得してISO8601形式の文字列に変換(例: "2025-01-15T10:30:45.123456")
        timestamp = datetime.now().isoformat()

        # DynamoDBに保存するアイテム(レコード)を作成
        item = {
            # プライマリキー: 日付とタイムスタンプを組み合わせてユニークなIDを生成
            # (例: "2025-01-15#2025-01-15T10:30:45.123456")
            'id': f"{date}#{timestamp}",
            'date': date,          # 日記の日付
            'title': title,        # 日記のタイトル
            'content': content,    # 日記の内容
            'createdAt': timestamp # 作成日時(タイムスタンプ)
        }

        # DynamoDBテーブルにアイテムを保存
        # put_item(): 新しいアイテムを追加、または既存のアイテムを上書き
        table.put_item(Item=item)

        # ========================================
        # ステップ4: 成功レスポンスを返す
        # ========================================
        return {
            'statusCode': 200,  # HTTPステータスコード: 200 OK(成功)
            'headers': {
                'Content-Type': 'application/json',
            },
            # 保存成功のメッセージと、保存したアイテムの内容を返す
            'body': json.dumps({
                'message': 'Data saved successfully',  # 成功メッセージ
                'item': item  # 保存したデータの内容
            })
        }

    except Exception as e:
        # ========================================
        # エラーハンドリング: 何か問題が発生した場合
        # ========================================
        # エラー内容をログに出力(CloudWatch Logsで確認可能)
        print(f"Error: {str(e)}")

        # エラーレスポンスを返す
        return {
            'statusCode': 500,  # HTTPステータスコード: 500 Internal Server Error
            'headers': {
                'Content-Type': 'application/json',
            },
            # エラー情報をJSON形式で返す
            'body': json.dumps({
                'error': 'Internal server error',  # エラーメッセージ
                'details': str(e)  # 詳細なエラー内容
            })
        }
