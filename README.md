# TrafficGuard AI

Dashboard phân tích video giao thông bằng model Roboflow `real-time-accident-detection/1`. Video gốc được phát ở trình duyệt; backend lấy mẫu frame theo cấu hình, inference bằng `inference-sdk`, rồi frontend vẽ prediction thật tại đúng timestamp đã phân tích. Không có API key thì hệ thống không chạy inference và không tự tạo detection.

## Yêu cầu

- Windows 10/11, Python 3.10–3.12 và Node.js 20+
- Roboflow API key có quyền dùng model nói trên

## Cài đặt trên Windows

Mở hai cửa sổ PowerShell tại thư mục dự án.

### Backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Điền API key vào `backend/.env`:

```dotenv
ROBOFLOW_API_KEY=your_actual_key
ROBOFLOW_MODEL_ID=real-time-accident-detection/1
ROBOFLOW_API_URL=https://serverless.roboflow.com
SAMPLE_FPS=4
CONFIDENCE_THRESHOLD=0.5
MAX_UPLOAD_MB=500
```

Chạy API:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

Tại cửa sổ PowerShell thứ hai, ở thư mục dự án:

```powershell
npm install
npm run dev
```

Mở URL Vite hiển thị trong terminal (thường là `http://localhost:5173`). Upload video; hệ thống tự gửi yêu cầu phân tích ngay sau khi upload thành công. Đợi kết quả rồi phát hoặc tua video. Các tình huống tai nạn nằm trong cột phải, cùng khu vực trạng thái phát hiện. Mỗi box chỉ xuất hiện quanh frame đã lấy mẫu; giữa các timestamp lấy mẫu, overlay được ẩn. Nhấn một sự kiện để tua tới thời điểm đầu tiên model phát hiện tai nạn.

## Cấu hình

- `SAMPLE_FPS`: tốc độ lấy mẫu từ 1 đến 10 FPS (mặc định 4).
- `CONFIDENCE_THRESHOLD`: ngưỡng confidence từ 0 đến 1 (mặc định 0.5).
- `MAX_UPLOAD_MB`: giới hạn dung lượng upload (mặc định 500 MB).
- `VITE_API_URL`: URL backend tùy chọn; mặc định dùng `/api` trên cùng địa chỉ frontend. Vite dev/preview chuyển tiếp `/api` tới `http://127.0.0.1:8000`. Khi triển khai bản build, cấu hình reverse proxy tương ứng hoặc đặt `VITE_API_URL` lúc build.

Các kết quả phân tích hoàn tất được lưu dưới dạng JSON tại `backend/analysis_cache` (không có database). File video upload được lưu tạm tại `backend/uploads`. Có thể xóa video upload để dọn dung lượng mà vẫn giữ kết quả; upload lại đúng nội dung video để dùng cache. Xóa `analysis_cache` nếu muốn xóa toàn bộ kết quả đã lưu.

## Chuẩn bị video để trình diễn nhanh

Phân tích trước tối đa 20 video trong một thư mục bằng Roboflow thật:

```powershell
cd backend
python precompute.py "D:\Videos\Demo" --limit 20
```

Lần chạy này cần mạng và sử dụng lượt inference Roboflow. Công cụ xử lý tuần tự; chạy lại sẽ bỏ qua video đã có kết quả hợp lệ, nên có thể tiếp tục sau khi gián đoạn. Thêm `--force` để chạy mới. Công cụ không sửa video nguồn.

Khi trình diễn: upload đúng video đã chuẩn bị; hệ thống tự nạp kết quả đã lưu và cho phép phát ngay khi nạp xong. Dashboard ghi rõ khi đang dùng kết quả Roboflow đã lưu. Đổi tên file vẫn dùng được cache; sửa/cắt/nén lại video, đổi model, sampling FPS hoặc confidence threshold sẽ cần phân tích lại. Video chưa có cache vẫn chạy inference thật. Nếu gặp lỗi, nút **Thử lại** gửi lại yêu cầu phân tích. Để bỏ qua cache và gọi API mới có thể dùng `POST /api/analyze/{video_id}?force=true` hoặc chạy công cụ chuẩn bị với `--force`.

Cache giữ nguyên prediction, confidence, bounding box và timestamp từ inference; không sinh detection hoặc tiến trình giả. Cache dùng được sau khi khởi động lại backend. Vẫn cần thời gian upload và đọc JSON; không cam kết mọi video nạp tức thì. Giữ nguyên thư mục `analysis_cache` khi di chuyển bản demo sang máy khác.

## Toàn bộ 50 video và kiểm tra class

```powershell
cd backend
python precompute.py "D:\Download\Positive_Vidoes-20261111T204154Z-1-001\Positive_Vidoes" --limit 50
python audit_results.py "D:\Download\Positive_Vidoes-20261111T204154Z-1-001\Positive_Vidoes"
```

Lệnh đầu tái sử dụng kết quả đã có và phân tích các video còn thiếu. Lệnh sau tạo báo cáo `analysis_cache/demo_manifest.json` gồm nhãn thực tế, confidence cao nhất theo loại, các frame có confidence giữa hai loại gần nhau và số frame tai nạn ở các ngưỡng 0.3/0.5/0.7/0.9. Đây là thống kê đầu ra model, không phải phép đo accuracy.

Hệ thống đang dùng model Roboflow có sẵn, không huấn luyện lại trọng số. Muốn hiệu chỉnh threshold theo precision/recall hoặc fine-tune model cần gán nhãn đúng loại và khoảng thời gian tai nạn cho mỗi video, cùng tập kiểm tra độc lập. File `ground_truth.template.json` được tạo để bắt đầu gán nhãn; `incidents: null` nghĩa là chưa gán nhãn, không có nghĩa là Normal. Mỗi incident nên gồm `class_id`, `start`, `end` (giây). Không tự hạ ngưỡng chỉ để mọi video đều báo tai nạn.

Giao diện chỉ báo class/confidence của frame đang xem; cửa sổ gộp event 3 giây không kéo dài cảnh báo trên các frame Normal. Nếu nhiều loại cùng xuất hiện, cảnh báo chính lấy confidence cao nhất, các loại còn lại vẫn được hiển thị.

Thống kê các ngưỡng chỉ dùng prediction API đã trả về, không thể phục hồi detection bị lọc ở inference server.

## Kiểm thử và build

Kiểm tra đồng bộ timestamp phía frontend (Node.js 24+): `node --test src/sampling.test.mjs`.

```powershell
cd backend
pytest -q
```

```powershell
cd ..
npm run build
```

Backend kiểm tra nhanh tại `http://localhost:8000/api/health`. Trường `api_key_configured` cho biết có cấu hình key; nó không xác minh key/model cho đến khi chạy inference thật. Lỗi Roboflow được hiển thị trạng thái `error` (không bị đổi thành Normal).
