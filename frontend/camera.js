// DOM要素の取得
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const preview = document.getElementById('preview');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const startCameraBtn = document.getElementById('startCamera');
const captureBtn = document.getElementById('capture');
const retakeBtn = document.getElementById('retake');
const uploadBtn = document.getElementById('upload');
const apiUrlInput = document.getElementById('apiUrl');
const statusDiv = document.getElementById('status');

// グローバル変数
let stream = null;
let capturedBlob = null;

// ローカルストレージからAPI URLを読み込む
window.addEventListener('DOMContentLoaded', () => {
    const savedApiUrl = localStorage.getItem('presignedUrlApiUrl');
    if (savedApiUrl) {
        apiUrlInput.value = savedApiUrl;
    }
});

// API URLの保存
apiUrlInput.addEventListener('change', () => {
    localStorage.setItem('presignedUrlApiUrl', apiUrlInput.value);
});

// カメラ起動
startCameraBtn.addEventListener('click', async () => {
    try {
        showStatus('カメラを起動しています...', 'info');

        // MediaDevices APIの対応チェック
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('MediaDevices API非対応:', {
                navigator: !!navigator,
                mediaDevices: !!navigator.mediaDevices,
                getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
            });
            showStatus(
                `❌ このブラウザはカメラAPIに対応していません\n` +
                `Chrome、Safari、Firefox等のモダンブラウザをご利用ください。\n` +
                `また、HTTPSでアクセスしているか確認してください。`,
                'error'
            );
            return;
        }

        // カメラストリームを取得（背面カメラを優先）
        const constraints = {
            video: {
                facingMode: 'environment', // 背面カメラ
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            },
            audio: false
        };

        console.log('カメラ起動リクエスト:', constraints);
        stream = await navigator.mediaDevices.getUserMedia(constraints);

        console.log('カメラストリーム取得成功:', {
            tracks: stream.getTracks().map(track => ({
                kind: track.kind,
                label: track.label,
                enabled: track.enabled,
                readyState: track.readyState,
                settings: track.getSettings()
            }))
        });

        video.srcObject = stream;
        cameraPlaceholder.style.display = 'none';
        video.style.display = 'block';
        preview.style.display = 'none';
        captureBtn.disabled = false;
        retakeBtn.style.display = 'none';
        uploadBtn.disabled = true;

        // ビデオのメタデータが読み込まれるのを待つ
        video.onloadedmetadata = () => {
            console.log('ビデオメタデータ読み込み完了:', {
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                duration: video.duration
            });
        };

        showStatus('✅ カメラが起動しました！撮影ボタンを押してください。', 'success');

    } catch (error) {
        let errorMessage = '';
        const errorDetails = {
            name: error.name,
            message: error.message,
            constraint: error.constraint
        };

        console.error('カメラの起動に失敗:', errorDetails);

        // エラーの種類に応じた詳細メッセージ
        switch (error.name) {
            case 'NotAllowedError':
            case 'PermissionDeniedError':
                errorMessage =
                    `❌ カメラの使用が許可されていません\n` +
                    `ブラウザの設定でカメラへのアクセスを許可してください。\n` +
                    `🔒 アドレスバーのカメラアイコンをクリックして許可してください。`;
                break;
            case 'NotFoundError':
            case 'DevicesNotFoundError':
                errorMessage =
                    `❌ カメラが見つかりません\n` +
                    `デバイスにカメラが接続されているか確認してください。`;
                break;
            case 'NotReadableError':
            case 'TrackStartError':
                errorMessage =
                    `❌ カメラが使用中です\n` +
                    `他のアプリケーションがカメラを使用していないか確認してください。`;
                break;
            case 'OverconstrainedError':
                errorMessage =
                    `❌ カメラの設定が適切ではありません\n` +
                    `制約: ${error.constraint}\n` +
                    `カメラが要求された解像度に対応していない可能性があります。`;
                break;
            case 'SecurityError':
                errorMessage =
                    `❌ セキュリティエラー\n` +
                    `HTTPSでアクセスしているか確認してください。\n` +
                    `localhostまたはHTTPS接続が必要です。`;
                break;
            default:
                errorMessage =
                    `❌ カメラの起動に失敗しました\n` +
                    `エラー: ${error.name}\n` +
                    `詳細: ${error.message}\n` +
                    `コンソールで詳細を確認してください。`;
        }

        showStatus(errorMessage, 'error');
    }
});

// 撮影
captureBtn.addEventListener('click', () => {
    try {
        // ビデオの状態をチェック
        if (!video.videoWidth || !video.videoHeight) {
            const errorDetails = {
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                readyState: video.readyState,
                networkState: video.networkState,
                paused: video.paused,
                ended: video.ended
            };
            console.error('ビデオが準備できていません:', errorDetails);
            showStatus(
                `❌ ビデオが準備できていません\n` +
                `解像度: ${video.videoWidth}x${video.videoHeight}\n` +
                `ReadyState: ${video.readyState}\n` +
                `カメラを再起動してください。`,
                'error'
            );
            return;
        }

        // Canvasにビデオフレームを描画
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        console.log('Canvas設定:', {
            width: canvas.width,
            height: canvas.height,
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight
        });

        const context = canvas.getContext('2d');
        if (!context) {
            console.error('Canvas 2Dコンテキストの取得に失敗');
            showStatus('❌ Canvas 2Dコンテキストの取得に失敗しました。ブラウザが対応していない可能性があります。', 'error');
            return;
        }

        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        console.log('Canvas描画完了');

        // Canvasから画像をBlobとして取得
        canvas.toBlob((blob) => {
            if (!blob) {
                const errorDetails = {
                    canvasWidth: canvas.width,
                    canvasHeight: canvas.height,
                    contextType: typeof context
                };
                console.error('Blob生成に失敗:', errorDetails);
                showStatus(
                    `❌ 画像Blob生成に失敗しました\n` +
                    `Canvas: ${canvas.width}x${canvas.height}\n` +
                    `ブラウザのメモリ不足の可能性があります。`,
                    'error'
                );
                return;
            }

            console.log('Blob生成成功:', {
                size: blob.size,
                type: blob.type,
                sizeKB: (blob.size / 1024).toFixed(2) + ' KB'
            });

            capturedBlob = blob;

            // プレビュー表示
            try {
                const url = URL.createObjectURL(blob);
                preview.src = url;
                cameraPlaceholder.style.display = 'none';
                preview.style.display = 'block';
                video.style.display = 'none';

                // カメラストリームを停止
                if (stream) {
                    stream.getTracks().forEach(track => {
                        console.log('トラック停止:', {
                            kind: track.kind,
                            label: track.label,
                            readyState: track.readyState
                        });
                        track.stop();
                    });
                }

                // ボタンの状態を更新
                captureBtn.disabled = true;
                retakeBtn.style.display = 'inline-block';
                uploadBtn.disabled = false;

                showStatus(
                    `✅ 撮影しました！\n` +
                    `サイズ: ${(blob.size / 1024).toFixed(2)} KB\n` +
                    `解像度: ${canvas.width}x${canvas.height}\n` +
                    `S3にアップロードできます。`,
                    'success'
                );
            } catch (previewError) {
                console.error('プレビュー表示エラー:', previewError);
                showStatus(
                    `⚠️ 撮影は成功しましたが、プレビュー表示に失敗しました\n` +
                    `エラー: ${previewError.message}\n` +
                    `アップロードは可能です。`,
                    'error'
                );
            }
        }, 'image/jpeg', 0.95);

    } catch (error) {
        const errorDetails = {
            name: error.name,
            message: error.message,
            stack: error.stack,
            videoState: {
                width: video.videoWidth,
                height: video.videoHeight,
                readyState: video.readyState
            },
            canvasState: {
                width: canvas.width,
                height: canvas.height
            }
        };
        console.error('撮影に失敗:', errorDetails);

        showStatus(
            `❌ 撮影に失敗しました\n` +
            `エラー種別: ${error.name}\n` +
            `詳細: ${error.message}\n` +
            `コンソールで詳細を確認してください。`,
            'error'
        );
    }
});

// 再撮影
retakeBtn.addEventListener('click', () => {
    capturedBlob = null;
    preview.style.display = 'none';
    cameraPlaceholder.style.display = 'none';
    retakeBtn.style.display = 'none';
    uploadBtn.disabled = true;
    startCameraBtn.click(); // カメラを再起動
});

// S3にアップロード
uploadBtn.addEventListener('click', async () => {
    if (!capturedBlob) {
        showStatus('❌ 画像が撮影されていません。', 'error');
        return;
    }

    const apiUrl = apiUrlInput.value.trim();
    if (!apiUrl) {
        showStatus('❌ Lambda Function URLを入力してください。', 'error');
        return;
    }

    try {
        uploadBtn.disabled = true;
        showStatus('署名付きURLを取得しています...', 'info');

        console.log('API呼び出し開始:', {
            apiUrl: apiUrl,
            blobSize: capturedBlob.size,
            blobType: capturedBlob.type
        });

        // 1. 署名付きURLを取得
        const requestBody = JSON.stringify({
            file_extension: 'jpg'
        });

        console.log('リクエストボディ:', requestBody);

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: requestBody
        });

        console.log('API レスポンス:', {
            status: response.status,
            statusText: response.statusText,
            headers: Object.fromEntries(response.headers.entries())
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('API エラーレスポンス:', errorText);

            let errorData;
            try {
                errorData = JSON.parse(errorText);
            } catch {
                errorData = { message: errorText };
            }

            throw new Error(
                `API呼び出しエラー\n` +
                `ステータス: ${response.status} ${response.statusText}\n` +
                `詳細: ${errorData.message || errorText}`
            );
        }

        const data = await response.json();
        console.log('署名付きURL取得成功:', {
            upload_url: data.upload_url?.substring(0, 100) + '...',
            file_key: data.file_key,
            bucket_name: data.bucket_name
        });

        if (!data.upload_url) {
            throw new Error('署名付きURLが返されませんでした。APIレスポンスを確認してください。');
        }

        showStatus('S3に画像をアップロードしています...', 'info');

        // 2. 署名付きURLを使用してS3に画像をアップロード
        console.log('S3アップロード開始:', {
            url: data.upload_url.substring(0, 100) + '...',
            blobSize: capturedBlob.size,
        });

        // 署名付きURLにはContent-Typeが含まれているため、ヘッダーは不要
        // ヘッダーを指定するとプリフライトリクエストが発生し、CORSエラーになる
        const uploadResponse = await fetch(data.upload_url, {
            method: 'PUT',
            body: capturedBlob,
            headers: {
                'Content-Type': capturedBlob.type
            }
        });

        console.log('S3 レスポンス:', {
            status: uploadResponse.status,
            statusText: uploadResponse.statusText,
            headers: Object.fromEntries(uploadResponse.headers.entries())
        });

        if (!uploadResponse.ok) {
            const errorText = await uploadResponse.text();
            console.error('S3 エラーレスポンス:', errorText);

            throw new Error(
                `S3アップロードエラー\n` +
                `ステータス: ${uploadResponse.status} ${uploadResponse.statusText}\n` +
                `詳細: ${errorText || '不明なエラー'}\n` +
                `署名付きURLの有効期限が切れている可能性があります。`
            );
        }

        console.log('アップロード完全成功:', {
            bucket: data.bucket_name,
            key: data.file_key,
            size: capturedBlob.size
        });

        // 成功メッセージ
        showStatus(
            `✅ アップロード成功！\n` +
            `バケット: ${data.bucket_name}\n` +
            `キー: ${data.file_key}\n` +
            `サイズ: ${(capturedBlob.size / 1024).toFixed(2)} KB`,
            'success'
        );

        // ボタンの状態をリセット
        uploadBtn.disabled = true;
        capturedBlob = null;

    } catch (error) {
        const errorDetails = {
            name: error.name,
            message: error.message,
            stack: error.stack
        };

        console.error('アップロードに失敗:', errorDetails);

        let errorMessage = '';

        // ネットワークエラーの判定
        if (error.message.includes('Failed to fetch') || error.name === 'TypeError') {
            errorMessage =
                `❌ ネットワークエラー\n` +
                `API URLが正しいか確認してください。\n` +
                `CORS設定が適切か確認してください。\n` +
                `詳細: ${error.message}`;
        } else if (error.message.includes('API呼び出しエラー')) {
            errorMessage =
                `❌ Lambda関数エラー\n` +
                `${error.message}\n` +
                `Lambda Function URLを確認してください。`;
        } else if (error.message.includes('S3アップロードエラー')) {
            errorMessage =
                `❌ S3アップロードエラー\n` +
                `${error.message}`;
        } else {
            errorMessage =
                `❌ アップロードに失敗しました\n` +
                `エラー: ${error.name}\n` +
                `詳細: ${error.message}\n` +
                `コンソールで詳細を確認してください。`;
        }

        showStatus(errorMessage, 'error');
        uploadBtn.disabled = false;
    }
});

// ステータス表示
function showStatus(message, type) {
    statusDiv.textContent = message;

    // Bootstrapのアラートクラスをマッピング
    const alertTypes = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'info': 'alert-info'
    };

    statusDiv.className = `alert ${alertTypes[type] || 'alert-info'} status-box`;
    statusDiv.classList.remove('d-none');

    // 成功メッセージは5秒後に自動で消す
    if (type === 'success') {
        setTimeout(() => {
            statusDiv.classList.add('d-none');
        }, 5000);
    }
}

// ページを離れる前にカメラを停止
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
