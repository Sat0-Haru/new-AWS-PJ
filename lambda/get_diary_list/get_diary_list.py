# JSON形式のデータを扱うための標準ライブラリ
import json
# AWSサービスを操作するためのPython SDK
import boto3
# 環境変数を扱うための標準ライブラリ
import os

# DynamoDBを操作するためのリソースオブジェクトを作成
dynamodb = boto3.resource('dynamodb')

# 環境変数からDynamoDBテーブル名を取得
# この環境変数はCDKスタックで設定される
table_name = os.environ['TABLE_NAME']

# 指定されたテーブルに接続
table = dynamodb.Table(table_name)

def handler(event, context):
    """
    Lambda関数のメイン処理

    【機能】
    DynamoDBテーブルをスキャンして、保存されている全ての日記データを取得する

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
        # ステップ1: DynamoDBテーブルをスキャン
        # ========================================
        # scan()メソッドでテーブル内の全データを取得
        # 注意: 大量のデータがある場合はページネーション処理が必要
        response = table.scan()

        # ========================================
        # ステップ2: 取得したアイテムを整形
        # ========================================
        # responseの'Items'キーに取得したデータのリストが格納されている
        items = response.get('Items', [])

        # createdAtフィールドで降順ソート(新しい日記が先頭に来るように)
        # lambda関数を使って、各アイテムのcreatedAtフィールドを基準にソート
        # reverse=Trueで降順(新しいものが先)
        sorted_items = sorted(items, key=lambda x: x.get('createdAt', ''), reverse=True)

        # ========================================
        # ステップ3: 成功レスポンスを返す
        # ========================================
        return {
            'statusCode': 200,  # HTTPステータスコード: 200 OK(成功)
            'headers': {
                'Content-Type': 'application/json',
            },
            # 取得した日記一覧をJSON形式で返す
            'body': json.dumps({
                'items': sorted_items,  # ソート済みの日記リスト
                'count': len(sorted_items)  # 取得した日記の件数
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
