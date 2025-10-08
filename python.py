import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        # Sử dụng ValueError để xử lý trong khối try-except bên ngoài
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Tài chính TỰ ĐỘNG (Chức năng 5 cũ) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File và Xử lý Dữ liệu ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo biến để lưu trữ DataFrame đã xử lý (để dùng trong Chat và các phần khác)
df_processed = None
thanh_toan_hien_hanh_N_1 = "N/A"
thanh_toan_hien_hanh_N = "N/A"

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, kiểm tra chia cho 0
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else 0
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                st.warning("Nợ Ngắn hạn bằng 0, không thể tính Chỉ số Thanh toán Hiện hành.")
            
            # --- Chức năng 5: Nhận xét AI Tự động ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (đảm bảo các giá trị đã được tính toán ở trên)
            try:
                tsnh_growth = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]
            except IndexError:
                tsnh_growth = "N/A"
                
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{tsnh_growth:.2f}%" if isinstance(tsnh_growth, (int, float)) else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else "N/A", 
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, (int, float)) else "N/A"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY", None) 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
    

# ======================================================================================
# --- THÊM KHUNG CHAT HỎI ĐÁP VỚI GEMINI (Chức năng mới) ---
# ======================================================================================

# Phân tách nội dung chính và sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Trợ lý Tài chính Gemini Chat 🤖")

# 1. Khởi tạo Chat History trong session_state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Chào bạn! Tôi là Trợ lý AI Phân tích Tài chính. Bạn có câu hỏi nào về các chỉ số, phương pháp phân tích, hoặc cần giải thích **dữ liệu đã tải lên** không?"}
    ]

# 2. Hàm gọi API Gemini cho Chat
def chat_with_gemini_conversational(prompt, api_key, history, processed_data=None):
    """Gửi tin nhắn và lịch sử trò chuyện đến Gemini API và nhận phản hồi, có kèm ngữ cảnh dữ liệu."""
    try:
        if not api_key:
            return "Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets."

        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Xây dựng nội dung (contents) bao gồm System Instruction và History
        
        # 1. System Instruction (Định nghĩa vai trò và cung cấp dữ liệu ngữ cảnh)
        system_parts = [
            {
                "text": "Bạn là một Trợ lý AI Phân tích Tài chính thân thiện. Hãy trả lời các câu hỏi một cách ngắn gọn, chuyên nghiệp và hữu ích. Cố gắng tham chiếu dữ liệu tài chính đã cung cấp (nếu có) để đưa ra câu trả lời cụ thể hơn. Tuyệt đối không giả định thông tin tài chính nào khác ngoài dữ liệu được cung cấp."
            }
        ]
        
        if processed_data is not None:
            # Gửi toàn bộ dữ liệu đã xử lý dưới dạng markdown cho mô hình
            # Sử dụng to_markdown(index=False) cho dễ đọc
            context_data = f"\n\nDữ liệu tài chính hiện tại của người dùng đã được xử lý:\n{processed_data.to_markdown(index=False)}\n\n"
            system_parts.append({"text": context_data})

        # Cấu trúc nội dung cho API
        contents = []

        # Thêm System Instruction
        # Trong thư viện google-genai, việc định nghĩa vai trò trong contents
        # đã tự động áp dụng System Instruction nếu không dùng Conversation service
        # Cách sau đây là cách truyền ngữ cảnh và lịch sử trong 1 lần gọi
        
        # Thêm System Instruction (sử dụng như tin nhắn đầu tiên)
        contents.append({
            "role": "user", # Đặt System Instruction vào vai trò 'user' đầu tiên để truyền ngữ cảnh
            "parts": system_parts
        })


        # 2. Thêm lịch sử trò chuyện (chuyển đổi định dạng Streamlit sang định dạng Gemini)
        # Bỏ qua tin nhắn chào mừng ban đầu của assistant khi xây dựng lịch sử chính thức
        # Lịch sử chat được xây dựng từ tin nhắn thứ 2 trở đi trong st.session_state["messages"]
        chat_history_for_api = st.session_state["messages"][1:]

        for message in chat_history_for_api:
            # Chỉ thêm các tin nhắn có vai trò 'user' hoặc 'assistant'
            if message["role"] in ["user", "assistant"]: 
                contents.append({"role": message["role"], "parts": [{"text": message["content"]}]})
        
        # 3. Thêm prompt hiện tại của người dùng (nếu nó chưa được thêm vào history trước đó)
        # Vì prompt đã được thêm vào st.session_state["messages"] trước khi gọi hàm này, 
        # nó đã nằm trong chat_history_for_api, nên ta không cần thêm lại.

        # Gọi API
        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong quá trình trò chuyện: {e}"


# Hiển thị tất cả tin nhắn trong st.session_state (trong sidebar)

# Lấy history đã lọc (chỉ user/assistant) để hiển thị
# Tin nhắn system context không được hiển thị (tức tin nhắn chào mừng là tin nhắn 0)
display_history = [m for m in st.session_state["messages"] if m["role"] in ["user", "assistant"]]

for message in display_history:
    # Sử dụng st.sidebar.chat_message
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input từ người dùng (luôn đặt ngoài container để nó ở cuối sidebar)
if prompt := st.sidebar.chat_input("Hỏi Gemini về báo cáo này..."):
    
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    
    # 1. Thêm tin nhắn người dùng vào lịch sử
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # 2. Hiển thị tin nhắn người dùng mới nhất
    with st.sidebar.chat_message("user"):
        st.markdown(prompt)

    # 3. Gọi API để nhận phản hồi của AI
    with st.sidebar.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            
            # Gửi toàn bộ lịch sử (bao gồm cả prompt hiện tại)
            ai_response = chat_with_gemini_conversational(
                prompt, 
                api_key, 
                st.session_state["messages"], # Gửi toàn bộ lịch sử
                df_processed # Gửi DataFrame đã xử lý (có thể là None nếu chưa upload file)
            )
            st.markdown(ai_response)
    
    # 4. Thêm tin nhắn AI vào lịch sử
    st.session_state["messages"].append({"role": "assistant", "content": ai_response})
    # Sau khi xử lý xong, Streamlit sẽ tự động rerun và hiển thị lại toàn bộ lịch sử
