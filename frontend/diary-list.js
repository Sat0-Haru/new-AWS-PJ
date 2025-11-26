// ========================================
// API エンドポイントの設定
// ========================================
// TODO: CDKデプロイ後、コンソールに出力される実際のLambda Function URLに置き換える
// 日記一覧を取得するAPIのURL
const getDiaryListApiUrl = 'GetDiaryListFunctionUrl';

// ========================================
// 日記一覧を取得して表示する関数
// ========================================
async function loadDiaryList() {
    // ローディング表示を表示
    document.getElementById('loading').classList.remove('d-none');
    // エラー表示を非表示
    document.getElementById('error').classList.add('d-none');
    // 日記一覧エリアをクリア
    document.getElementById('diaryList').innerHTML = '';
    // 空の状態表示を非表示
    document.getElementById('emptyState').classList.add('d-none');

    try {
        // ========================================
        // API 呼び出し: 日記一覧を取得
        // ========================================
        // fetch(): HTTPリクエストを送信する非同期関数
        // await: 非同期処理の完了を待つ
        const response = await fetch(getDiaryListApiUrl, {
            method: 'GET',  // HTTPメソッド: GET(データを取得)
        });

        // レスポンスのボディ(JSON文字列)をJavaScriptオブジェクトに変換
        const result = await response.json();

        // ローディング表示を非表示
        document.getElementById('loading').classList.add('d-none');

        // ========================================
        // レスポンスの処理
        // ========================================
        if (response.ok) {
            // 成功した場合(HTTPステータスコード: 200-299)

            // 日記データの配列を取得
            const diaries = result.items || [];

            if (diaries.length === 0) {
                // 日記が0件の場合、空の状態表示を表示
                document.getElementById('emptyState').classList.remove('d-none');
            } else {
                // 日記が1件以上ある場合、一覧を表示
                displayDiaryList(diaries);
            }
        } else {
            // 失敗した場合(HTTPステータスコード: 400, 500など)
            showError('日記の取得に失敗しました: ' + (result.error || '不明なエラー'));
        }
    } catch (error) {
        // ネットワークエラーやJSON解析エラーなど、予期しないエラーが発生した場合
        console.error('Error loading diary list:', error);

        // ローディング表示を非表示
        document.getElementById('loading').classList.add('d-none');

        // エラーメッセージを表示
        showError('エラーが発生しました: ' + error.message);
    }
}

// ========================================
// 日記一覧を画面に表示する関数
// ========================================
function displayDiaryList(diaries) {
    const diaryListElement = document.getElementById('diaryList');

    // 各日記データをカード形式で表示
    diaries.forEach((diary, index) => {
        // 日記カードを作成
        const diaryCard = createDiaryCard(diary, index);
        // 日記一覧エリアに追加
        diaryListElement.appendChild(diaryCard);
    });
}

// ========================================
// 日記カード要素を作成する関数
// ========================================
function createDiaryCard(diary, index) {
    // カード全体のdiv要素を作成
    const cardDiv = document.createElement('div');
    cardDiv.className = 'card mb-3';  // mb-3: margin-bottom(下の余白)を3段階分追加

    // カードボディのdiv要素を作成
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';

    // ========================================
    // カードヘッダー部分: 日付とタイトル
    // ========================================
    const headerDiv = document.createElement('div');
    headerDiv.className = 'd-flex justify-content-between align-items-start mb-2';

    // タイトル部分
    const titleDiv = document.createElement('div');
    titleDiv.className = 'flex-grow-1';  // flex-grow-1: 利用可能なスペースを埋める

    // 日付を表示
    const dateElement = document.createElement('small');
    dateElement.className = 'text-muted d-block';  // text-muted: グレーのテキスト, d-block: ブロック要素として表示
    dateElement.textContent = formatDate(diary.date);  // 日付をフォーマット
    titleDiv.appendChild(dateElement);

    // タイトルを表示
    const titleElement = document.createElement('h5');
    titleElement.className = 'card-title mb-0';
    titleElement.textContent = diary.title || '(タイトルなし)';
    titleDiv.appendChild(titleElement);

    headerDiv.appendChild(titleDiv);

    // バッジ(通し番号)を表示
    const badgeElement = document.createElement('span');
    badgeElement.className = 'badge bg-secondary';  // bg-secondary: グレーの背景色
    badgeElement.textContent = `#${index + 1}`;  // 1から始まる通し番号
    headerDiv.appendChild(badgeElement);

    cardBody.appendChild(headerDiv);

    // ========================================
    // カード本文部分: 日記の内容
    // ========================================
    const contentElement = document.createElement('p');
    contentElement.className = 'card-text';

    // 内容が長い場合は省略して表示(最初の200文字のみ)
    const content = diary.content || '(内容なし)';
    if (content.length > 200) {
        contentElement.textContent = content.substring(0, 200) + '...';
    } else {
        contentElement.textContent = content;
    }

    cardBody.appendChild(contentElement);

    // ========================================
    // カードフッター部分: 作成日時
    // ========================================
    const footerElement = document.createElement('small');
    footerElement.className = 'text-muted';
    footerElement.textContent = `作成日時: ${formatDateTime(diary.createdAt)}`;
    cardBody.appendChild(footerElement);

    // カードボディをカードに追加
    cardDiv.appendChild(cardBody);

    return cardDiv;
}

// ========================================
// 日付をフォーマットする関数
// ========================================
function formatDate(dateString) {
    if (!dateString) return '日付不明';

    try {
        // "2025-01-15" -> "2025年1月15日"
        const date = new Date(dateString + 'T00:00:00');  // タイムゾーン問題を回避
        const year = date.getFullYear();
        const month = date.getMonth() + 1;  // 月は0から始まるため+1
        const day = date.getDate();

        return `${year}年${month}月${day}日`;
    } catch (error) {
        return dateString;  // フォーマット失敗時は元の文字列を返す
    }
}

// ========================================
// 日時をフォーマットする関数
// ========================================
function formatDateTime(dateTimeString) {
    if (!dateTimeString) return '不明';

    try {
        // "2025-01-15T10:30:45.123456" -> "2025年1月15日 10:30"
        const dateTime = new Date(dateTimeString);
        const year = dateTime.getFullYear();
        const month = dateTime.getMonth() + 1;
        const day = dateTime.getDate();
        const hours = String(dateTime.getHours()).padStart(2, '0');  // 2桁にゼロ埋め
        const minutes = String(dateTime.getMinutes()).padStart(2, '0');

        return `${year}年${month}月${day}日 ${hours}:${minutes}`;
    } catch (error) {
        return dateTimeString;  // フォーマット失敗時は元の文字列を返す
    }
}

// ========================================
// エラーメッセージを表示する関数
// ========================================
function showError(message) {
    const errorElement = document.getElementById('error');
    errorElement.textContent = message;
    errorElement.classList.remove('d-none');  // エラー表示を表示
}

// ========================================
// ページ読み込み時の処理
// ========================================
// DOMContentLoaded: HTMLの読み込みが完了したときに発生するイベント
document.addEventListener('DOMContentLoaded', function() {
    // ページが読み込まれたら自動的に日記一覧を取得
    loadDiaryList();
});
