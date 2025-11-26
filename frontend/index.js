// ========================================
// API エンドポイントの設定
// ========================================
// TODO: CDKデプロイ後、コンソールに出力される実際のLambda Function URLに置き換える
// DynamoDBにデータを保存するAPIのURL
const apiUrl = 'SaveToDynamoDBFunctionUrl';

// TODO: CDKデプロイ後、コンソールに出力される実際のLambda Function URLに置き換える
// BedrockでAI日記を生成するAPIのURL
const generateApiUrl = 'GenerateDiaryFunctionUrl';

// ========================================
// AI で内容を生成ボタンのイベント処理
// ========================================
// 「AIで内容を生成」ボタンがクリックされたときの処理
// getElementById(): HTML要素をIDで取得
// addEventListener(): イベント(クリックなど)が発生したときに実行する関数を登録
document.getElementById('generateContentBtn').addEventListener('click', async function () {
    // タイトル入力欄の値を取得し、前後の空白を削除
    const title = document.getElementById('title').value.trim();

    // ========================================
    // バリデーション: タイトルが入力されているかチェック
    // ========================================
    if (!title) {
        // タイトルが空の場合、警告メッセージを表示
        document.getElementById('result').innerHTML =
            '<div class="alert alert-warning">タイトルを入力してください。</div>';
        return; // 処理を中断
    }

    // ========================================
    // UI の更新: ボタンを無効化してローディング表示
    // ========================================
    const btn = this;  // クリックされたボタン要素を参照
    const originalText = btn.textContent;  // ボタンの元のテキストを保存
    btn.disabled = true;  // ボタンを無効化(連続クリックを防止)
    btn.textContent = '生成中...';  // ボタンのテキストを変更

    try {
        // ========================================
        // API 呼び出し: BedrockでAI日記を生成
        // ========================================
        // fetch(): HTTPリクエストを送信する非同期関数
        // await: 非同期処理の完了を待つ
        const response = await fetch(generateApiUrl, {
            method: 'POST',  // HTTPメソッド: POST(データを送信)
            headers: {
                'Content-Type': 'application/json',  // 送信データの形式をJSON指定
            },
            // JSON.stringify(): JavaScriptオブジェクトをJSON文字列に変換
            body: JSON.stringify({ title: title })
        });

        // レスポンスのボディ(JSON文字列)をJavaScriptオブジェクトに変換
        const result = await response.json();

        // ========================================
        // レスポンスの処理
        // ========================================
        if (response.ok) {
            // 成功した場合(HTTPステータスコード: 200-299)

            // 生成された日記の内容をテキストエリアに自動入力
            document.getElementById('content').value = result.content;

            // 成功メッセージを表示
            document.getElementById('result').innerHTML =
                '<div class="alert alert-success">内容を生成しました!</div>';
        } else {
            // 失敗した場合(HTTPステータスコード: 400, 500など)

            // エラーメッセージを表示
            document.getElementById('result').innerHTML =
                '<div class="alert alert-danger">エラー: ' + (result.error || '内容の生成に失敗しました') + '</div>';
        }
    } catch (error) {
        // ネットワークエラーやJSON解析エラーなど、予期しないエラーが発生した場合
        document.getElementById('result').innerHTML =
            '<div class="alert alert-danger">エラー: ' + error.message + '</div>';
    } finally {
        // try-catchの成功・失敗に関わらず必ず実行される処理

        // ボタンを元の状態に戻す(再度クリック可能にする)
        btn.disabled = false;
        btn.textContent = originalText;
    }
});

// ========================================
// フォーム送信イベント処理
// ========================================
// 日記保存フォームが送信されたときの処理
// 'submit': フォームが送信されたときに発生するイベント
document.getElementById('dataForm').addEventListener('submit', async function (e) {
    // デフォルトのフォーム送信動作(ページリロード)を防止
    e.preventDefault();

    // ========================================
    // フォームからデータを取得
    // ========================================
    const date = document.getElementById('date').value;      // 日付
    const title = document.getElementById('title').value;    // タイトル
    const content = document.getElementById('content').value; // 内容

    // 送信するデータをオブジェクトにまとめる
    const data = {
        date: date,
        title: title,
        content: content
    };

    // ========================================
    // UI の更新: 送信ボタンを無効化してローディング表示
    // ========================================
    const submitBtn = this.querySelector('button[type="submit"]');  // 送信ボタン要素を取得
    const originalText = submitBtn.textContent;  // ボタンの元のテキストを保存
    submitBtn.disabled = true;  // ボタンを無効化(連続送信を防止)
    submitBtn.textContent = '送信中...';  // ボタンのテキストを変更

    try {
        // ========================================
        // API 呼び出し: DynamoDBにデータを保存
        // ========================================
        const response = await fetch(apiUrl, {
            method: 'POST',  // HTTPメソッド: POST(データを送信)
            headers: {
                'Content-Type': 'application/json',  // 送信データの形式をJSON指定
            },
            // データオブジェクトをJSON文字列に変換して送信
            body: JSON.stringify(data)
        });

        // レスポンスのボディ(JSON文字列)をJavaScriptオブジェクトに変換
        const result = await response.json();

        // ========================================
        // レスポンスの処理
        // ========================================
        if (response.ok) {
            // 成功した場合(HTTPステータスコード: 200-299)

            // 成功メッセージを表示
            document.getElementById('result').innerHTML =
                '<div class="alert alert-success">データが正常に保存されました!</div>';

            // フォームをリセット(すべての入力欄をクリア)
            document.getElementById('dataForm').reset();

            // 日付フィールドを今日の日付に再設定
            document.getElementById('date').valueAsDate = new Date();
        } else {
            // 失敗した場合(HTTPステータスコード: 400, 500など)

            // エラーメッセージを表示
            document.getElementById('result').innerHTML =
                '<div class="alert alert-danger">エラー: ' + (result.error || 'データの保存に失敗しました') + '</div>';
        }
    } catch (error) {
        // ネットワークエラーやJSON解析エラーなど、予期しないエラーが発生した場合
        document.getElementById('result').innerHTML =
            '<div class="alert alert-danger">エラー: ' + error.message + '</div>';
    } finally {
        // try-catchの成功・失敗に関わらず必ず実行される処理

        // ボタンを元の状態に戻す(再度送信可能にする)
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
});
