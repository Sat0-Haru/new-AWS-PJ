# AWS CDK 日記アプリケーション

このプロジェクトは、AWS CDK を使用して構築された日記アプリケーションです。AI（Amazon Bedrock）を使って日記の内容を自動生成し、DynamoDB に保存できます。

## 📁 ディレクトリ構成

```
aws_project_2025_template/
├── bin/                              # CDKアプリケーションのエントリーポイント
│   └── aws_project_2025_template.ts  # CDKアプリを初期化するファイル
│
├── lib/                              # インフラストラクチャのコード（IaC）
│   └── aws_project_2025_template-stack.ts  # AWSリソースを定義するメインファイル
│                                     # S3、Lambda、DynamoDBなどのリソースをここで作成
│
├── frontend/                         # Webアプリケーションのフロントエンド
│   ├── index.html                    # 日記入力フォームのHTMLファイル
│   ├── index.js                      # 日記作成ページのJavaScript（API呼び出しなど）
│   ├── diary-list.html               # 日記一覧ページのHTMLファイル
│   ├── diary-list.js                 # 日記一覧ページのJavaScript
│   ├── camera.html                   # カメラ撮影ページのHTMLファイル
│   ├── camera.js                     # カメラ撮影ページのJavaScript
│   ├── architecture.html             # アーキテクチャ説明ページ
│   ├── error.html                    # エラーページ
│   └── favicon.ico                   # ファビコン
│
├── lambda/                           # Lambda関数のコード
│   ├── save_to_dynamodb/             # DynamoDBにデータを保存する関数
│   │   ├── save_to_dynamodb.py       # データ保存のメイン処理
│   │   └── requirements.txt          # Python依存パッケージ
│   │
│   ├── generate_diary_content/       # AIで日記の内容を生成する関数
│   │   ├── generate_diary_content.py # Bedrockを使った内容生成処理
│   │   └── requirements.txt          # Python依存パッケージ
│   │
│   ├── get_diary_list/               # 日記一覧を取得する関数
│   │   ├── get_diary_list.py         # 日記一覧取得のメイン処理
│   │   └── requirements.txt          # Python依存パッケージ
│   │
│   └── generate_presigned_url/       # S3署名付きURLを生成する関数
│       ├── generate_presigned_url.py # 署名付きURL生成のメイン処理
│       └── requirements.txt          # Python依存パッケージ
│
├── test/                             # テストコード
│   └── aws_project_2025_template.test.ts  # CDKスタックのテスト
│
├── .devcontainer/                    # 開発コンテナ設定
│   └── devcontainer.json             # VS Code Dev Container設定ファイル
│
├── .gitignore                        # Gitで追跡しないファイルを指定
├── .npmignore                        # npmパッケージから除外するファイルを指定
├── cdk.json                          # CDKの設定ファイル
├── package.json                      # Node.jsプロジェクトの依存関係とスクリプト
├── package-lock.json                 # 依存関係のバージョンロックファイル
├── tsconfig.json                     # TypeScriptの設定ファイル
├── jest.config.js                    # Jestテストフレームワークの設定
└── README.md                         # このファイル
```

## 🔍 各ディレクトリの詳細説明

### 📂 `bin/` - アプリケーションのエントリーポイント
- **役割**: CDK アプリケーションを起動する最初のファイルが入っています
- **解説**: CDK が最初に読み込むファイルです

### 📂 `lib/` - インフラストラクチャ定義
- **役割**: AWS のリソース（S3、Lambda、DynamoDB など）を TypeScript で定義します
- **解説**:
  - ここがプロジェクトの心臓部です
  - `aws_project_2025_template-stack.ts` を編集して、AWS リソースを追加・変更します
  - コードを書くだけで、実際の AWS リソースが作成されます（Infrastructure as Code）

### 📂 `frontend/` - ユーザーが見る画面
- **役割**: ブラウザで表示される HTML と JavaScript が入っています
- **解説**:
  - `index.html`: 日記を入力するフォーム画面
  - `app.js`: ボタンを押したときの動作（API 呼び出しなど）
  - このフォルダの内容は S3 バケットにアップロードされ、静的ウェブサイトとして公開されます

### 📂 `lambda/` - サーバーレス関数
- **役割**: AWS Lambda で実行される Python のコードが入っています
- **解説**:
  - 各サブディレクトリが 1 つの Lambda 関数に対応
  - `save_to_dynamodb/`: 日記を DynamoDB に保存する関数
  - `generate_diary_content/`: AI（Bedrock）で日記の内容を生成する関数
  - `requirements.txt`: その関数で使う Python ライブラリを指定

### 📂 `test/` - テストコード
- **役割**: インフラストラクチャが正しく定義されているかテストします
- **解説**: `npm test` でテストを実行できます

## 🚀 使い方

### 0. 前提条件
- AWS アカウントを作成済み
- AWS CLI がインストール済み
- Node.js がインストール済み

### 1. AWS 認証情報の設定

デプロイする前に、AWS の認証情報を設定する必要があります。

#### Step 1-1: AWS CLI で認証情報を設定

ターミナルで以下のコマンドを実行：

```bash
aws configure
```

以下の情報を順番に入力：

```
AWS Access Key ID [None]: <アクセスキーIDを入力>
AWS Secret Access Key [None]: <シークレットアクセスキーを入力>
Default region name [None]: ap-northeast-1
Default output format [None]: json
```

**各項目の説明:**
- **AWS Access Key ID**: IAM で取得したアクセスキー ID
- **AWS Secret Access Key**: IAM で取得したシークレットアクセスキー
- **Default region name**: 使用する AWS リージョン（東京リージョンは `ap-northeast-1`）
- **Default output format**: 出力形式（`json` を推奨）

#### Step 1-2: 認証情報の確認

正しく設定できたか確認：

```bash
aws sts get-caller-identity
```

以下のような出力が表示されれば成功です：
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

### 2. 依存関係のインストール
```bash
npm install
```

### 3. デプロイ（AWS にリソースを作成）
```bash
npx cdk deploy
```

### 4. デプロイ後の作業
デプロイが完了すると、以下の情報が表示されます：
- `WebsiteURL`: フロントエンドのURL
- `SaveToDynamoDBFunctionUrl`: データ保存APIのURL
- `GenerateDiaryFunctionUrl`: AI内容生成APIのURL
- `GetDiaryListFunctionUrl`: 日記一覧取得APIのURL

#### Step 4-1: 日記作成ページのAPIエンドポイント設定
`frontend/index.js` の以下の部分を、表示されたURLに置き換えてください：
```javascript
const apiUrl = 'SaveToDynamoDBFunctionUrl';
const generateApiUrl = 'GenerateDiaryFunctionUrl';
```

#### Step 4-2: 日記一覧ページのAPIエンドポイント設定
`frontend/diary-list.js` の以下の部分を、表示されたURLに置き換えてください：
```javascript
const getDiaryListApiUrl = 'GetDiaryListFunctionUrl';
```

#### Step 4-3: 変更を反映
再度デプロイして変更を反映：
```bash
npx cdk deploy
```

### 5. リソースの削除
```bash
npx cdk destroy
```

## 📚 主な AWS サービス

- **S3**: フロントエンドのホスティング
- **Lambda**: サーバーレス関数（Python）
- **DynamoDB**: データベース（NoSQL）
- **Bedrock**: AI サービス（日記の内容生成）
- **CDK**: インフラストラクチャをコードで管理

## 🎯 このプロジェクトで学べること

1. **Infrastructure as Code (IaC)**: コードでインフラを管理する方法
2. **サーバーレスアーキテクチャ**: サーバー管理不要で動くシステム
3. **AWS の基本サービス**: S3、Lambda、DynamoDB の使い方
4. **AI 統合**: Amazon Bedrock を使った生成 AI の活用

## 💡 よくある質問

**Q: `npx cdk deploy` で何が起こるの？**
A: `lib/` に書いたコードが実際の AWS リソースに変換され、自動的に作成されます。

**Q: Lambda 関数を追加するには？**
A: 1) `lambda/` に新しいフォルダを作成、2) Python コードを書く、3) `lib/aws_project_2025_template-stack.ts` で Lambda リソースを定義

**Q: フロントエンドを変更するには？**
A: `frontend/index.html` や `frontend/app.js` を編集して、`npx cdk deploy` で再デプロイ

**Q: お金はかかる？**
A: AWS の無料利用枠内であれば無料ですが、使いすぎると課金されます。不要なリソースは `npx cdk destroy` で削除しましょう。

**Q: `aws configure` で設定した認証情報はどこに保存される？**
A: `~/.aws/credentials` ファイルに保存されます。このファイルには機密情報が含まれるため、Git にコミットしないよう注意してください。

**Q: 認証情報を間違えて入力してしまった場合は？**
A: もう一度 `aws configure` を実行すれば、上書きできます。または、`~/.aws/credentials` ファイルを直接編集することもできます。
