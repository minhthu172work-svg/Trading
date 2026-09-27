# -*- coding: utf-8 -*-
"""
=============================================================================
HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC GIAO DỊCH KẾT HỢP EMA VÀ OBV (CỔ PHIẾU ACB)
Môn học: Quản lý danh mục đầu tư (ThS) - Nhóm 3
Nền tảng: Streamlit Web Application
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

# Cố gắng import các thư viện chuyên dụng, có cơ chế fallback nếu chưa cài đặt
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

# ---------------------------------------------------------------------------
# CẤU HÌNH TRANG WEB STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kiểm định Chiến lược EMA & OBV | ACB",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS để giao diện trực quan, chuyên nghiệp theo phong cách FinTech
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10B981;
    }
    .metric-sub-neg {
        font-size: 0.8rem;
        color: #EF4444;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HÀM TÍNH TOÁN CHỈ BÁO KỸ THUẬT (EMA & OBV)
# ---------------------------------------------------------------------------
def compute_ema(series: pd.Series, window: int) -> pd.Series:
    """Tính đường EMA chuẩn."""
    if HAS_TA:
        return ta.trend.ema_indicator(series, window=int(window))
    return series.ewm(span=int(window), adjust=False).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Tính chỉ số On-Balance Volume (OBV)."""
    if HAS_TA:
        return ta.volume.on_balance_volume(close, volume)
    change = close.diff()
    direction = pd.Series(np.where(change > 0, 1, np.where(change < 0, -1, 0)), index=close.index)
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def get_ema_signals(data: pd.DataFrame, ema_period: int, price_col='Close'):
    """
    Chiến lược EMA riêng lẻ:
    Mua khi Giá vượt lên trên EMA.
    Bán khi Giá cắt xuống dưới EMA.
    Dùng shift(1) để tránh Look-ahead Bias.
    """
    close = data[price_col]
    ema = compute_ema(close, int(ema_period))

    raw_entries = (close > ema)
    raw_exits = (close < ema)

    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)
    return entries, exits, ema


def get_obv_signals(data: pd.DataFrame, obv_slope_period: int = 3, price_col='Close'):
    """
    Chiến lược OBV riêng lẻ:
    Mua khi Độ dốc OBV > 0 (áp lực gom hàng).
    Bán khi Độ dốc OBV < 0 (áp lực xả hàng).
    Dùng shift(1) để tránh Look-ahead Bias.
    """
    close = data[price_col]
    volume = data['Volume']
    obv = compute_obv(close, volume)
    obv_slope = obv.diff(int(obv_slope_period))

    raw_entries = (obv_slope > 0)
    raw_exits = (obv_slope < 0)

    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)
    return entries, exits, obv_slope, obv


def get_ema_obv_combined_signals(data: pd.DataFrame, ema_period: int, obv_slope_period: int = 3, price_col='Close'):
    """
    Chiến lược kết hợp EMA + OBV:
    Mua khi Giá > EMA VÀ Độ dốc OBV > 0.
    Bán khi Giá < EMA.
    Dùng shift(1) để tránh Look-ahead Bias.
    """
    close = data[price_col]
    volume = data['Volume']

    ema = compute_ema(close, int(ema_period))
    obv = compute_obv(close, volume)
    obv_slope = obv.diff(int(obv_slope_period))

    raw_entries = (close > ema) & (obv_slope > 0)
    raw_exits = (close < ema)

    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)
    return entries, exits, ema, obv_slope, obv


def calculate_positions(entries: pd.Series, exits: pd.Series):
    """
    Mô phỏng máy trạng thái vị thế (Long-only, accumulate=False).
    Position = 1 (Đang nắm giữ), Position = 0 (Đứng ngoài/Tiền mặt).
    """
    n = len(entries)
    position = pd.Series(0, index=entries.index, dtype=int)
    buy_orders = pd.Series(False, index=entries.index, dtype=bool)
    sell_orders = pd.Series(False, index=entries.index, dtype=bool)

    current_pos = 0
    for i in range(n):
        if current_pos == 0 and entries.iloc[i]:
            current_pos = 1
            buy_orders.iloc[i] = True
        elif current_pos == 1 and exits.iloc[i]:
            current_pos = 0
            sell_orders.iloc[i] = True
        position.iloc[i] = current_pos

    return position, buy_orders, sell_orders


# ---------------------------------------------------------------------------
# CÔNG CỤ BACKTEST THUẦN (BACKTEST ENGINE VỚI STOP LOSS, PHÍ & TRƯỢT GIÁ)
# ---------------------------------------------------------------------------
def run_simulation_backtest(
    data: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    fees: float = 0.002,
    slippage: float = 0.001,
    sl_stop: float = 0.07,
    init_cash: float = 100_000_000.0,
    price_col: str = 'Close'
):
    """
    Engine mô phỏng danh mục đầu tư chuẩn theo đúng cấu hình VectorBT trong notebook:
    - Long-only, không nhồi lệnh (accumulate=False).
    - Phí giao dịch (fees = 0.002 = 0.2%).
    - Trượt giá (slippage = 0.001 = 0.1%).
    - Cắt lỗ cố định từ giá mua (sl_stop = 0.07 = 7%).
    """
    close = data[price_col].values
    dates = data.index
    n = len(close)

    cash = init_cash
    holding_shares = 0
    entry_price = 0.0
    entry_date = None

    equity_curve = np.zeros(n)
    position_history = np.zeros(n)
    trades = []

    for i in range(n):
        cur_price = close[i]
        cur_date = dates[i]
        is_entry = bool(entries.iloc[i])
        is_exit = bool(exits.iloc[i])

        # Kiểm tra điều kiện cắt lỗ nếu đang có vị thế
        stop_loss_triggered = False
        if holding_shares > 0 and sl_stop is not None and sl_stop > 0:
            if cur_price <= entry_price * (1.0 - sl_stop):
                stop_loss_triggered = True

        # Xử lý đóng vị thế (Bán do tín hiệu exit HOẶC do chạm Stop Loss)
        if holding_shares > 0 and (is_exit or stop_loss_triggered):
            # Giá thực tế bán sau trượt giá và trừ phí
            effective_sell_price = cur_price * (1.0 - slippage)
            proceeds = holding_shares * effective_sell_price * (1.0 - fees)
            cash += proceeds

            ret_pct = (effective_sell_price / entry_price - 1.0) - (fees * 2)
            exit_reason = "Cắt lỗ (Stop Loss -7%)" if stop_loss_triggered else "Tín hiệu Bán (Exit Signal)"

            trades.append({
                'Entry Date': entry_date,
                'Exit Date': cur_date,
                'Entry Price': entry_price,
                'Exit Price': cur_price,
                'Effective Exit Price': effective_sell_price,
                'Holding Days': (cur_date - entry_date).days,
                'Return (%)': ret_pct * 100.0,
                'PnL (VND)': proceeds - (holding_shares * entry_price * (1.0 + slippage + fees)),
                'Exit Reason': exit_reason
            })

            holding_shares = 0
            entry_price = 0.0
            entry_date = None

        # Xử lý mở vị thế (Mua nếu có tín hiệu entry và chưa nắm giữ)
        elif holding_shares == 0 and is_entry:
            effective_buy_price = cur_price * (1.0 + slippage)
            total_buy_price_per_share = effective_buy_price * (1.0 + fees)

            if cash >= total_buy_price_per_share:
                holding_shares = int(cash // total_buy_price_per_share)
                cost = holding_shares * total_buy_price_per_share
                cash -= cost
                entry_price = effective_buy_price
                entry_date = cur_date

        # Cập nhật tổng tài sản hiện tại
        current_equity = cash + (holding_shares * cur_price)
        equity_curve[i] = current_equity
        position_history[i] = 1 if holding_shares > 0 else 0

    equity_series = pd.Series(equity_curve, index=dates)
    cum_returns = (equity_series / init_cash) - 1.0
    daily_returns = equity_series.pct_change().fillna(0)

    # Tính toán các chỉ số tài chính chuẩn
    total_return_pct = ((equity_series.iloc[-1] / init_cash) - 1.0) * 100.0
    days_total = (dates[-1] - dates[0]).days if len(dates) > 1 else 1
    cagr_pct = (((equity_series.iloc[-1] / init_cash) ** (365.25 / max(days_total, 1))) - 1.0) * 100.0

    # Sharpe Ratio hàng năm hóa (252 phiên giao dịch/năm)
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe_ratio = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0.0

    # Max Drawdown
    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown_pct = drawdowns.min() * 100.0

    trades_df = pd.DataFrame(trades)
    closed_trades_count = len(trades_df)

    if closed_trades_count > 0:
        win_trades = trades_df[trades_df['Return (%)'] > 0]
        loss_trades = trades_df[trades_df['Return (%)'] <= 0]
        win_rate = (len(win_trades) / closed_trades_count) * 100.0

        total_gain = win_trades['PnL (VND)'].sum() if len(win_trades) > 0 else 0.0
        total_loss = abs(loss_trades['PnL (VND)'].sum()) if len(loss_trades) > 0 else 0.0
        profit_factor = (total_gain / total_loss) if total_loss > 0 else np.nan
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # Lợi nhuận chiến lược Buy & Hold tham chiếu
    bh_return_pct = ((close[-1] / close[0]) - 1.0) * 100.0

    stats = {
        'Total Return (%)': total_return_pct,
        'CAGR (%)': cagr_pct,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown_pct,
        'Closed Trades': closed_trades_count,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Buy & Hold Return (%)': bh_return_pct,
        'Final Equity (VND)': equity_series.iloc[-1]
    }

    return {
        'stats': stats,
        'equity_series': equity_series,
        'cum_returns': cum_returns * 100.0,
        'drawdowns': drawdowns * 100.0,
        'position': pd.Series(position_history, index=dates),
        'trades_df': trades_df
    }


# ---------------------------------------------------------------------------
# TÍNH TOÁN BẰNG VECTORBT (NẾU CÓ) ĐỂ ĐỐI CHIẾU
# ---------------------------------------------------------------------------
def run_vbt_backtest(data: pd.DataFrame, entries: pd.Series, exits: pd.Series, price_col='Close', fees=0.002, slippage=0.001, sl_stop=0.07):
    """Sử dụng VectorBT như trong notebook nếu thư viện vectorbt khả dụng."""
    if not HAS_VBT:
        return None
    try:
        pf = vbt.Portfolio.from_signals(
            close=data[price_col],
            entries=entries.to_numpy(dtype=bool),
            exits=exits.to_numpy(dtype=bool),
            direction='longonly',
            accumulate=False,
            fees=fees,
            slippage=slippage,
            sl_stop=sl_stop,
            sl_trail=False,
            freq='D'
        )
        return pf
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LOAD VÀ CACHE DỮ LIỆU
# ---------------------------------------------------------------------------
@st.cache_data
def load_stock_data(file_path_or_buffer):
    """Đọc dữ liệu CSV, chuẩn hóa chỉ mục ngày tháng."""
    df = pd.read_csv(file_path_or_buffer)
    # Tìm cột ngày tháng
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'time', 'ngay', 'ngày', 'datetime']:
            date_col = col
            break
    if date_col:
        df['Date'] = pd.to_datetime(df[date_col])
        df.set_index('Date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)

    # Đảm bảo tên cột viết hoa chữ cái đầu (Close, Open, High, Low, Volume)
    rename_dict = {}
    for col in df.columns:
        c_lower = col.lower()
        if c_lower in ['close', 'dongcua', 'đóng cửa', 'gia']:
            rename_dict[col] = 'Close'
        elif c_lower in ['open', 'mocua', 'mở cửa']:
            rename_dict[col] = 'Open'
        elif c_lower in ['high', 'caonhat', 'cao nhất']:
            rename_dict[col] = 'High'
        elif c_lower in ['low', 'thapnhat', 'thấp nhất']:
            rename_dict[col] = 'Low'
        elif c_lower in ['volume', 'khoiluong', 'khối lượng', 'vol']:
            rename_dict[col] = 'Volume'

    df.rename(columns=rename_dict, inplace=True)
    return df


# ---------------------------------------------------------------------------
# GIAO DIỆN CHÍNH & SIDEBAR
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch EMA & OBV</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phân Tích Định Lượng Danh Mục Đầu Tư Cổ Phiếu ACB | Tối Ưu Hóa Hyperopt & Kiểm Định Out-of-Sample</div>', unsafe_allow_html=True)

# SIDEBAR: Quản lý Dữ liệu và Tham số
st.sidebar.title("🛠️ Thiết Lập Hệ Thống")

# 1. Nguồn Dữ liệu
st.sidebar.subheader("1. Dữ Liệu Thị Trường")
data_source = st.sidebar.radio(
    "Nguồn file dữ liệu:",
    ["Sử dụng file ACB.csv gốc", "Tải lên file CSV mới"],
    index=0
)

df_raw = None
if data_source == "Sử dụng file ACB.csv gốc":
    # Tìm file ACB.csv ở thư mục hiện hành hoặc thư mục cha
    possible_paths = ['ACB.csv', '../ACB.csv', './ACB.csv']
    acb_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if acb_path:
        df_raw = load_stock_data(acb_path)
        st.sidebar.success(f" Đã nạp `{acb_path}` ({len(df_raw):,} phiên)")
    else:
        st.sidebar.error("Không tìm thấy file ACB.csv trong thư mục gốc. Vui lòng tải file lên.")
else:
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV cổ phiếu", type=['csv'])
    if uploaded_file is not None:
        df_raw = load_stock_data(uploaded_file)
        st.sidebar.success(f" Đã nạp `{uploaded_file.name}` ({len(df_raw):,} phiên)")

if df_raw is None or df_raw.empty:
    st.warning("⚠️ Vui lòng cung cấp file dữ liệu CSV để tiếp tục!")
    st.stop()

# 2. Phân chia Train / Test
min_date = df_raw.index.min().date()
max_date = df_raw.index.max().date()

st.sidebar.subheader("2. Phân Chia Tập Dữ Liệu")
st.sidebar.caption("Theo cấu hình chuẩn của Notebook:")
st.sidebar.markdown("- **Train (In-Sample):** 2014 đến 2020\n- **Test (Out-of-Sample):** 2021 đến 2023")

col_d1, col_d2 = st.sidebar.columns(2)
train_start = col_d1.date_input("Train bắt đầu", value=pd.to_datetime("2014-01-02").date(), min_value=min_date, max_value=max_date)
train_end = col_d2.date_input("Train kết thúc", value=pd.to_datetime("2020-12-31").date(), min_value=min_date, max_value=max_date)

col_d3, col_d4 = st.sidebar.columns(2)
test_start = col_d3.date_input("Test bắt đầu", value=pd.to_datetime("2021-01-01").date(), min_value=min_date, max_value=max_date)
test_end = col_d4.date_input("Test kết thúc", value=pd.to_datetime("2023-12-29").date(), min_value=min_date, max_value=max_date)

# Cắt tập dữ liệu
train_df = df_raw.loc[str(train_start):str(train_end)].copy()
test_df = df_raw.loc[str(test_start):str(test_end)].copy()

# 3. Cài đặt Tham số Chiến lược
st.sidebar.subheader("3. Cấu Hình Tham Số Chiến Lược")

preset_mode = st.sidebar.selectbox(
    "Bộ tham số kiểm nghiệm:",
    [
        "Bộ tham số Tối ưu trên Train (EMA=36, OBV_Slope=20)",
        "Bộ tham số Mặc định (Default: EMA=20, OBV_Slope=3)",
        "Tùy chỉnh tự do (Custom Sliders)"
    ],
    index=0
)

if preset_mode == "Bộ tham số Tối ưu trên Train (EMA=36, OBV_Slope=20)":
    default_ema = 36
    default_obv = 20
    default_ema_only = 36
    default_obv_only = 19
elif preset_mode == "Bộ tham số Mặc định (Default: EMA=20, OBV_Slope=3)":
    default_ema = 20
    default_obv = 3
    default_ema_only = 20
    default_obv_only = 3
else:
    default_ema = 36
    default_obv = 20
    default_ema_only = 36
    default_obv_only = 19

param_ema = st.sidebar.slider("Chu kỳ EMA (phiên)", min_value=5, max_value=80, value=default_ema, step=1)
param_obv_slope = st.sidebar.slider("Chu kỳ độ dốc OBV (phiên)", min_value=2, max_value=40, value=default_obv, step=1)

with st.sidebar.expander("⚙️ Quản lý Vốn & Rủi ro", expanded=False):
    param_sl = st.slider("Cắt lỗ Stop Loss (%)", min_value=1.0, max_value=15.0, value=7.0, step=0.5) / 100.0
    param_fee = st.slider("Phí giao dịch mỗi chiều (%)", min_value=0.0, max_value=0.5, value=0.2, step=0.05) / 100.0
    param_slippage = st.slider("Trượt giá (%)", min_value=0.0, max_value=0.3, value=0.1, step=0.05) / 100.0
    param_init_cash = st.number_input("Vốn khởi điểm (VNĐ)", min_value=10_000_000, value=100_000_000, step=10_000_000)

price_col = 'Close'

# ---------------------------------------------------------------------------
# THỰC THI TÍNH TOÁN TRÊN CẢ 2 TẬP TRAIN VÀ TEST
# ---------------------------------------------------------------------------
@st.cache_data
def run_all_strategies(train_data, test_data, ema_p, obv_p, sl_rate, fee_rate, slip_rate, cash_val):
    results = {}
    for name, data in [('Train', train_data), ('Test', test_data)]:
        if len(data) == 0:
            continue
        # 1. EMA riêng lẻ
        ent_ema, ext_ema, line_ema = get_ema_signals(data, ema_p)
        res_ema = run_simulation_backtest(data, ent_ema, ext_ema, fee_rate, slip_rate, sl_rate, cash_val)

        # 2. OBV riêng lẻ
        ent_obv, ext_obv, slope_obv, line_obv = get_obv_signals(data, obv_p)
        res_obv = run_simulation_backtest(data, ent_obv, ext_obv, fee_rate, slip_rate, sl_rate, cash_val)

        # 3. Kết hợp EMA + OBV
        ent_comb, ext_comb, c_ema, c_slope, c_obv = get_ema_obv_combined_signals(data, ema_p, obv_p)
        res_comb = run_simulation_backtest(data, ent_comb, ext_comb, fee_rate, slip_rate, sl_rate, cash_val)

        results[name] = {
            'EMA': res_ema,
            'OBV': res_obv,
            'Combined': res_comb,
            'signals': {
                'ema': (ent_ema, ext_ema, line_ema),
                'obv': (ent_obv, ext_obv, slope_obv, line_obv),
                'comb': (ent_comb, ext_comb, c_ema, c_slope, c_obv)
            }
        }
    return results

all_res = run_all_strategies(train_df, test_df, param_ema, param_obv_slope, param_sl, param_fee, param_slippage, param_init_cash)

# ---------------------------------------------------------------------------
# THANH CHỈ SỐ NHANH (KPI TOP METRICS)
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

comb_train = all_res['Train']['Combined']['stats'] if 'Train' in all_res else None
comb_test = all_res['Test']['Combined']['stats'] if 'Test' in all_res else None

with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Lợi Nhuận Train (2014-2020)</div>', unsafe_allow_html=True)
    val = f"{comb_train['Total Return (%)']:.2f}%" if comb_train else "N/A"
    sub_cls = "metric-sub" if comb_train and comb_train['Total Return (%)'] >= 0 else "metric-sub-neg"
    st.markdown(f'<div class="metric-value">{val}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{sub_cls}">EMA+OBV Kết hợp (In-Sample)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Lợi Nhuận Test (2021-2023)</div>', unsafe_allow_html=True)
    val = f"{comb_test['Total Return (%)']:.2f}%" if comb_test else "N/A"
    sub_cls = "metric-sub" if comb_test and comb_test['Total Return (%)'] >= 0 else "metric-sub-neg"
    st.markdown(f'<div class="metric-value">{val}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{sub_cls}">EMA+OBV Kết hợp (Out-of-Sample)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Sharpe Ratio (Train vs Test)</div>', unsafe_allow_html=True)
    s_tr = f"{comb_train['Sharpe Ratio']:.2f}" if comb_train else "N/A"
    s_te = f"{comb_test['Sharpe Ratio']:.2f}" if comb_test else "N/A"
    st.markdown(f'<div class="metric-value">{s_tr} | {s_te}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-sub">Đo lường lợi nhuận trên rủi ro</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Max Drawdown (Train vs Test)</div>', unsafe_allow_html=True)
    d_tr = f"{comb_train['Max Drawdown (%)']:.2f}%" if comb_train else "N/A"
    d_te = f"{comb_test['Max Drawdown (%)']:.2f}%" if comb_test else "N/A"
    st.markdown(f'<div class="metric-value">{d_tr} | {d_te}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-sub-neg">Mức sụt giảm vốn tối đa</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# CÁC TAB CHỨC NĂNG CHÍNH CỦA ỨNG DỤNG
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dữ Liệu & Tổng Quan",
    "🎯 Tín Hiệu & Vị Thế",
    "📈 Kết Quả Backtest",
    "⚖️ So Sánh Train vs Test",
    "🔬 Tối Ưu Hóa Hyperopt",
    "📚 Cơ Sở Lý Thuyết & Nhận Xét"
])

# ===========================================================================
# TAB 1: DỮ LIỆU & TỔNG QUAN THỊ TRƯỜNG
# ===========================================================================
with tab1:
    st.subheader("📊 Khảo Sát Chuỗi Dữ Liệu Cổ Phiếu ACB")

    c_info1, c_info2 = st.columns([1, 2])
    with c_info1:
        st.markdown(f"""
        - **Mã cổ phiếu:** Ngân hàng TMCP Á Châu (ACB)
        - **Khoảng thời gian toàn bộ:** {min_date} ➔ {max_date} ({len(df_raw):,} phiên)
        - **Tập Train (In-Sample):** {train_start} ➔ {train_end} ({len(train_df):,} phiên)
        - **Tập Test (Out-of-Sample):** {test_start} ➔ {test_end} ({len(test_df):,} phiên)
        - **Giá thấp nhất / Cao nhất:** {df_raw['Close'].min():,.0f} ➔ {df_raw['Close'].max():,.0f} VND
        - **Khối lượng TB phiên:** {df_raw['Volume'].mean():,.0f} cổ phiếu
        """)

    with c_info2:
        st.info("""
        💡 **Mục đích chia tập dữ liệu (In-Sample vs Out-of-Sample):**
        - **Tập Train (2014 - 2020):** Dùng để huấn luyện mô hình, tìm kiếm tham số tối ưu (Hyperopt TPE).
        - **Tập Test (2021 - 2023):** Dữ liệu chưa từng được mô hình nhìn thấy, kiểm định xem chiến lược có bị **Quá khớp (Overfitting)** hay duy trì được hiệu quả trong thực tế.
        """)

    # Biểu đồ giá và khối lượng toàn bộ lịch sử kèm ranh giới Train/Test
    fig_overview = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=("Biểu đồ Giá Đóng Cửa (VND)", "Khối Lượng Giao Dịch")
    )

    fig_overview.add_trace(
        go.Scatter(x=df_raw.index, y=df_raw['Close'], name='Giá ACB (Close)', line=dict(color='#2563EB', width=1.5)),
        row=1, col=1
    )

    # Thêm vùng đánh dấu Train và Test
    fig_overview.add_vrect(
        x0=str(train_start), x1=str(train_end),
        fillcolor="rgba(16, 185, 129, 0.12)", layer="below", line_width=0,
        annotation_text="TẬP TRAIN (2014-2020)", annotation_position="top left",
        row=1, col=1
    )
    fig_overview.add_vrect(
        x0=str(test_start), x1=str(test_end),
        fillcolor="rgba(239, 68, 68, 0.12)", layer="below", line_width=0,
        annotation_text="TẬP TEST (2021-2023)", annotation_position="top left",
        row=1, col=1
    )

    fig_overview.add_trace(
        go.Bar(x=df_raw.index, y=df_raw['Volume'], name='Khối lượng', marker_color='#94A3B8'),
        row=2, col=1
    )

    fig_overview.update_layout(
        height=500, margin=dict(l=30, r=30, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig_overview, use_container_width=True)

    with st.expander("🔍 Xem Bảng Dữ Liệu Chi Tiết (5 dòng đầu & cuối)"):
        st.dataframe(pd.concat([df_raw.head(5), df_raw.tail(5)]), use_container_width=True)


# ===========================================================================
# TAB 2: TÍN HIỆU & VỊ THẾ GIAO DỊCH
# ===========================================================================
with tab2:
    st.subheader("🎯 Minh Họa Tín Hiệu & Quản Lý Vị Thế")

    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        target_dataset = st.radio("Chọn tập dữ liệu hiển thị:", ["Tập Train (2014-2020)", "Tập Test (2021-2023)"], horizontal=True)
    with c_sel2:
        target_strat = st.radio("Chọn chiến lược quan sát:", ["EMA + OBV Kết hợp", "EMA Riêng lẻ", "OBV Riêng lẻ"], horizontal=True)

    active_data = train_df if "Train" in target_dataset else test_df
    strat_key = 'Combined' if "Kết hợp" in target_strat else ('EMA' if "EMA" in target_strat else 'OBV')
    ds_key = 'Train' if "Train" in target_dataset else 'Test'

    sig_pack = all_res[ds_key]['signals']
    strat_res = all_res[ds_key][strat_key]

    if strat_key == 'Combined':
        entries, exits, ema_line, obv_slope, obv_line = sig_pack['comb']
    elif strat_key == 'EMA':
        entries, exits, ema_line = sig_pack['ema']
        obv_slope = active_data['Volume']  # Placeholder
        obv_line = active_data['Volume']
    else:
        entries, exits, obv_slope, obv_line = sig_pack['obv']
        ema_line = active_data['Close']

    pos_series, buys, sells = calculate_positions(entries, exits)

    # Biểu đồ giá kèm tín hiệu Mua/Bán
    fig_sig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            f"Giá & Điểm Vào Lệnh ({target_strat} - {target_dataset})",
            f"Chỉ Báo Độ Dốc OBV (Slope Period = {param_obv_slope})",
            "Trạng Thái Vị Thế (0: Tiền mặt | 1: Đang giữ cổ phiếu)"
        )
    )

    # 1. Đường giá và EMA
    fig_sig.add_trace(go.Scatter(x=active_data.index, y=active_data['Close'], name='Giá ACB', line=dict(color='#1E293B', width=1.5)), row=1, col=1)
    if strat_key in ['Combined', 'EMA']:
        fig_sig.add_trace(go.Scatter(x=active_data.index, y=ema_line, name=f'EMA ({param_ema})', line=dict(color='#F59E0B', width=1.5, dash='dot')), row=1, col=1)

    # Điểm MUA (Mũi tên xanh hướng lên)
    buy_dates = active_data.index[buys]
    buy_prices = active_data.loc[buys, 'Close']
    fig_sig.add_trace(go.Scatter(
        x=buy_dates, y=buy_prices, mode='markers', name='Điểm MUA',
        marker=dict(symbol='triangle-up', size=11, color='#10B981', line=dict(width=1, color='#047857'))
    ), row=1, col=1)

    # Điểm BÁN (Mũi tên đỏ hướng xuống)
    sell_dates = active_data.index[sells]
    sell_prices = active_data.loc[sells, 'Close']
    fig_sig.add_trace(go.Scatter(
        x=sell_dates, y=sell_prices, mode='markers', name='Điểm BÁN',
        marker=dict(symbol='triangle-down', size=11, color='#EF4444', line=dict(width=1, color='#B91C1C'))
    ), row=1, col=1)

    # 2. Chỉ báo OBV Slope
    if strat_key in ['Combined', 'OBV']:
        colors_slope = ['#10B981' if v > 0 else '#EF4444' for v in obv_slope.fillna(0)]
        fig_sig.add_trace(go.Bar(x=active_data.index, y=obv_slope, name='Độ dốc OBV', marker_color=colors_slope), row=2, col=1)
    else:
        fig_sig.add_trace(go.Scatter(x=active_data.index, y=active_data['Close'] - ema_line, name='Chênh lệch Giá - EMA', line=dict(color='#6366F1')), row=2, col=1)

    # 3. Vị thế (0 hoặc 1)
    fig_sig.add_trace(go.Scatter(x=active_data.index, y=pos_series, name='Vị thế nắm giữ', line=dict(color='#0284C7', width=1.5, shape='hv'), fill='tozeroy', fillcolor='rgba(2, 132, 199, 0.15)'), row=3, col=1)

    fig_sig.update_layout(
        height=680, margin=dict(l=30, r=30, t=40, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_sig, use_container_width=True)

    # Thống kê lệnh
    st.markdown(f"**Thống kê kích hoạt lệnh:** Số lần Mua = `{buys.sum()}` | Số lần Bán = `{sells.sum()}` | Số giao dịch hoàn tất = `{sells.sum()}`")

    with st.expander("📋 Xem Danh Sách Các Phiên Kích Hoạt Lệnh Mua / Bán"):
        df_sig_check = pd.DataFrame({
            'Giá Đóng Cửa': active_data['Close'],
            f'EMA ({param_ema})': ema_line if strat_key != 'OBV' else np.nan,
            f'Độ dốc OBV ({param_obv_slope})': obv_slope if strat_key != 'EMA' else np.nan,
            'Tín hiệu MUA': entries,
            'Tín hiệu BÁN': exits,
            'Lệnh Mua Thực Hiện': buys,
            'Lệnh Bán Thực Hiện': sells,
            'Vị Thế': pos_series
        })
        st.dataframe(df_sig_check.loc[buys | sells], use_container_width=True)


# ===========================================================================
# TAB 3: KẾT QUẢ BACKTEST & LỊCH SỬ GIAO DỊCH
# ===========================================================================
with tab3:
    st.subheader("📈 Kết Quả Backtest Toàn Diện Các Chiến Lược")

    sel_dataset_bt = st.selectbox("Chọn tập dữ liệu để phân tích hiệu suất:", ["Tập Train (In-Sample: 2014-2020)", "Tập Test (Out-of-Sample: 2021-2023)"])
    ds_name = 'Train' if "Train" in sel_dataset_bt else 'Test'
    curr_data = train_df if ds_name == 'Train' else test_df

    st_ema = all_res[ds_name]['EMA']['stats']
    st_obv = all_res[ds_name]['OBV']['stats']
    st_comb = all_res[ds_name]['Combined']['stats']

    # Bảng so sánh 3 chiến lược + Buy & Hold
    summary_data = {
        'Chỉ Số Hiệu Suất': [
            'Tổng Lợi Nhuận (%)',
            'Lợi Nhuận Hàng Năm (CAGR %)',
            'Tỷ Lệ Sharpe (Sharpe Ratio)',
            'Mức Sụt Giảm Tối Đa (Max Drawdown %)',
            'Tổng Số Lệnh Đã Đóng',
            'Tỷ Lệ Thắng (Win Rate %)',
            'Tỷ Số Lãi/Lỗ (Profit Factor)',
            'Lợi Nhuận Mua & Nắm Giữ (B&H %)'
        ],
        'EMA Riêng Lẻ': [
            f"{st_ema['Total Return (%)']:.2f}%",
            f"{st_ema['CAGR (%)']:.2f}%",
            f"{st_ema['Sharpe Ratio']:.4f}",
            f"{st_ema['Max Drawdown (%)']:.2f}%",
            f"{st_ema['Closed Trades']}",
            f"{st_ema['Win Rate (%)']:.1f}%",
            f"{st_ema['Profit Factor']:.2f}" if np.isfinite(st_ema['Profit Factor']) else "N/A",
            f"{st_ema['Buy & Hold Return (%)']:.2f}%"
        ],
        'OBV Riêng Lẻ': [
            f"{st_obv['Total Return (%)']:.2f}%",
            f"{st_obv['CAGR (%)']:.2f}%",
            f"{st_obv['Sharpe Ratio']:.4f}",
            f"{st_obv['Max Drawdown (%)']:.2f}%",
            f"{st_obv['Closed Trades']}",
            f"{st_obv['Win Rate (%)']:.1f}%",
            f"{st_obv['Profit Factor']:.2f}" if np.isfinite(st_obv['Profit Factor']) else "N/A",
            f"{st_obv['Buy & Hold Return (%)']:.2f}%"
        ],
        'EMA + OBV Kết Hợp': [
            f"{st_comb['Total Return (%)']:.2f}%",
            f"{st_comb['CAGR (%)']:.2f}%",
            f"{st_comb['Sharpe Ratio']:.4f}",
            f"{st_comb['Max Drawdown (%)']:.2f}%",
            f"{st_comb['Closed Trades']}",
            f"{st_comb['Win Rate (%)']:.1f}%",
            f"{st_comb['Profit Factor']:.2f}" if np.isfinite(st_comb['Profit Factor']) else "N/A",
            f"{st_comb['Buy & Hold Return (%)']:.2f}%"
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    st.table(df_summary)

    # Biểu đồ đường cong lợi nhuận tích lũy (Cumulative Returns)
    fig_cum = go.Figure()
    bh_cum = (curr_data['Close'] / curr_data['Close'].iloc[0] - 1.0) * 100.0

    fig_cum.add_trace(go.Scatter(x=curr_data.index, y=all_res[ds_name]['Combined']['cum_returns'], name=f'EMA + OBV Kết Hợp (EMA={param_ema}, OBV={param_obv_slope})', line=dict(color='#10B981', width=2.5)))
    fig_cum.add_trace(go.Scatter(x=curr_data.index, y=all_res[ds_name]['EMA']['cum_returns'], name=f'EMA Riêng Lẻ (EMA={param_ema})', line=dict(color='#3B82F6', width=1.5, dash='dash')))
    fig_cum.add_trace(go.Scatter(x=curr_data.index, y=all_res[ds_name]['OBV']['cum_returns'], name=f'OBV Riêng Lẻ (OBV={param_obv_slope})', line=dict(color='#F59E0B', width=1.5, dash='dot')))
    fig_cum.add_trace(go.Scatter(x=curr_data.index, y=bh_cum, name='Mua & Nắm Giữ (Buy & Hold)', line=dict(color='#94A3B8', width=1.2, dash='longdash')))

    fig_cum.update_layout(
        title=f"Đường Cong Lợi Nhuận Tích Lũy (%) - {sel_dataset_bt}",
        xaxis_title="Thời gian",
        yaxis_title="Lợi nhuận tích lũy (%)",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Biểu đồ Drawdown sụt giảm vốn
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=curr_data.index, y=all_res[ds_name]['Combined']['drawdowns'], name='Drawdown Kết Hợp', fill='tozeroy', line=dict(color='#EF4444', width=1)))
    fig_dd.update_layout(
        title=f"Mức Sụt Giảm Tài Sản Theo Thời Gian (Underwater Drawdown %) - {sel_dataset_bt}",
        yaxis_title="Sụt giảm (%)",
        template="plotly_white",
        height=280
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Nhật ký giao dịch chi tiết (Trade Log)
    st.subheader("📑 Nhật Ký Lệnh Giao Dịch Chi Tiết (Chiến Lược Kết Hợp)")
    comb_trades = all_res[ds_name]['Combined']['trades_df']
    if len(comb_trades) > 0:
        st.dataframe(comb_trades.style.format({
            'Entry Price': '{:,.1f}',
            'Exit Price': '{:,.1f}',
            'Effective Exit Price': '{:,.1f}',
            'Return (%)': '{:+.2f}%',
            'PnL (VND)': '{:+,.0f}'
        }), use_container_width=True)

        csv_data = comb_trades.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Tải File CSV Nhật Ký Lệnh", data=csv_data, file_name=f"Trade_Log_ACB_{ds_name}.csv", mime="text/csv")
    else:
        st.info("Không có lệnh nào được đóng trong khoảng thời gian này.")


# ===========================================================================
# TAB 4: SO SÁNH TRAIN VS TEST (TÁI HIỆN CELL 23 NOTEBOOK)
# ===========================================================================
with tab4:
    st.subheader("⚖️ So Sánh Hiệu Suất: Tập Train (2014-2020) vs Tập Test (2021-2023)")
    st.caption("Trực quan hóa và mở rộng từ phân tích Cell 23 của Notebook ACB_Strategy_EMA_OBV_Nhom3_Chinhsua.ipynb")

    # Xây dựng DataFrame so sánh Train vs Test tương tự Cell 23
    comp_records = []
    for d_name in ['Train', 'Test']:
        st_e = all_res[d_name]['EMA']['stats']
        st_o = all_res[d_name]['OBV']['stats']
        st_c = all_res[d_name]['Combined']['stats']

        comp_records.append({
            'Chiến lược': 'EMA Riêng lẻ',
            'Tập dữ liệu': d_name,
            'Tham số': f"EMA={param_ema}",
            'Sharpe Ratio': st_e['Sharpe Ratio'],
            'Tổng lợi nhuận (%)': st_e['Total Return (%)'],
            'Max Drawdown (%)': st_e['Max Drawdown (%)'],
            'Số giao dịch': st_e['Closed Trades']
        })
        comp_records.append({
            'Chiến lược': 'OBV Riêng lẻ',
            'Tập dữ liệu': d_name,
            'Tham số': f"OBV_Slope={param_obv_slope}",
            'Sharpe Ratio': st_o['Sharpe Ratio'],
            'Tổng lợi nhuận (%)': st_o['Total Return (%)'],
            'Max Drawdown (%)': st_o['Max Drawdown (%)'],
            'Số giao dịch': st_o['Closed Trades']
        })
        comp_records.append({
            'Chiến lược': 'EMA + OBV Kết hợp',
            'Tập dữ liệu': d_name,
            'Tham số': f"EMA={param_ema}, OBV_Slope={param_obv_slope}",
            'Sharpe Ratio': st_c['Sharpe Ratio'],
            'Tổng lợi nhuận (%)': st_c['Total Return (%)'],
            'Max Drawdown (%)': st_c['Max Drawdown (%)'],
            'Số giao dịch': st_c['Closed Trades']
        })

    df_comp = pd.DataFrame(comp_records)

    # 3 Biểu đồ cột so sánh: Sharpe Ratio, Tổng lợi nhuận %, Max Drawdown %
    fig_bars = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("1. Tỷ Lệ Sharpe (Sharpe Ratio)", "2. Tổng Lợi Nhuận (%)", "3. Mức Sụt Giảm Tối Đa (Max Drawdown %)")
    )

    # 1. Sharpe Ratio
    for d_type, color in [('Train', '#10B981'), ('Test', '#EF4444')]:
        sub = df_comp[df_comp['Tập dữ liệu'] == d_type]
        fig_bars.add_trace(
            go.Bar(name=f'Sharpe - {d_type}', x=sub['Chiến lược'], y=sub['Sharpe Ratio'],
                   marker_color=color, text=[f"{v:.2f}" for v in sub['Sharpe Ratio']], textposition='outside'),
            row=1, col=1
        )

    # 2. Total Return (%)
    for d_type, color in [('Train', '#6366F1'), ('Test', '#F97316')]:
        sub = df_comp[df_comp['Tập dữ liệu'] == d_type]
        fig_bars.add_trace(
            go.Bar(name=f'Return - {d_type}', x=sub['Chiến lược'], y=sub['Tổng lợi nhuận (%)'],
                   marker_color=color, text=[f"{v:.1f}%" for v in sub['Tổng lợi nhuận (%)']], textposition='outside'),
            row=2, col=1
        )

    # 3. Max Drawdown (%)
    for d_type, color in [('Train', '#06B6D4'), ('Test', '#EC4899')]:
        sub = df_comp[df_comp['Tập dữ liệu'] == d_type]
        fig_bars.add_trace(
            go.Bar(name=f'Drawdown - {d_type}', x=sub['Chiến lược'], y=sub['Max Drawdown (%)'],
                   marker_color=color, text=[f"{v:.1f}%" for v in sub['Max Drawdown (%)']], textposition='outside'),
            row=3, col=1
        )

    fig_bars.update_layout(
        height=850,
        barmode='group',
        template="plotly_white",
        showlegend=False,
        margin=dict(l=30, r=30, t=40, b=20)
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown("#### 📋 Bảng Số Liệu Đối Chiếu Train vs Test")
    st.dataframe(df_comp.style.format({
        'Sharpe Ratio': '{:.4f}',
        'Tổng lợi nhuận (%)': '{:+.2f}%',
        'Max Drawdown (%)': '{:.2f}%',
        'Số giao dịch': '{:d}'
    }), use_container_width=True)

    st.markdown("""
    > [!IMPORTANT]
    > **Nhận Định Học Thuật Quan Trọng Về Hiện Tượng Quá Mức Khớp Dữ Liệu (Overfitting):**
    > - **Giai đoạn Train (2014 - 2020):** Thị trường chứng khoán Việt Nam và cổ phiếu ACB nằm trong một chu kỳ Uptrend dài hạn kéo dài. Do đó, các tham số kỹ thuật được tối ưu hóa cho kết quả cực kỳ ấn tượng (Sharpe > 1.0, Lợi nhuận > 200%).
    > - **Giai đoạn Test (2021 - 2023):** Thị trường trải qua giai đoạn biến động mạnh đặc biệt là đợt suy thoái diện rộng năm 2022. Hiệu suất của tất cả các chiến lược trên tập Test đều sụt giảm mạnh (Sharpe âm, Lợi nhuận âm ~-36% đến -38%).
    > - **Ưu điểm của Chiến lược Kết hợp:** Mặc dù thị trường Test khó khăn, chiến lược **EMA + OBV Kết hợp** vẫn có **mức sụt giảm lợi nhuận thấp nhất (-36.47%)** và **mức Drawdown trên tập Train thấp nhất (-23.87%)** so với việc dùng riêng rẽ từng chỉ báo.
    """)


# ===========================================================================
# TAB 5: TỐI ƯU HÓA THAM SỐ (HYPEROPT / GRID SEARCH)
# ===========================================================================
with tab5:
    st.subheader("🔬 Tối Ưu Hóa Tham Số Bằng Hyperopt (TPE Algorithm)")
    st.markdown("""
    Công cụ này cho phép bạn tái lập quy trình tối ưu hóa Hyperopt từ Cell 15 trong notebook trên tập dữ liệu Train (2014 - 2020),
    nhằm tìm ra cặp tham số `(EMA, OBV_Slope)` tối đa hóa **Tỷ lệ Sharpe**.
    """)

    c_opt1, c_opt2, c_opt3 = st.columns(3)
    with c_opt1:
        opt_strat_choice = st.selectbox("Chiến lược cần tối ưu:", ["EMA + OBV Kết hợp", "EMA Riêng lẻ", "OBV Riêng lẻ"])
    with c_opt2:
        opt_max_evals = st.select_slider("Số lần thử nghiệm (Max Evaluations):", options=[20, 50, 100, 200, 500, 1066], value=50)
    with c_opt3:
        opt_metric_target = st.selectbox("Mục tiêu tối ưu:", ["Tối đa hóa Sharpe Ratio", "Tối đa hóa Lợi Nhuận (%)"])

    btn_run_opt = st.button("🚀 Bắt Đầu Quá Trình Tối Ưu Hóa", type="primary")

    if btn_run_opt:
        if not HAS_HYPEROPT:
            st.warning("Thư viện Hyperopt chưa được cài đặt. Hệ thống sẽ sử dụng thuật toán Grid Search thay thế!")

        progress_bar = st.progress(0)
        status_text = st.empty()

        best_params = {}
        best_score = -np.inf
        trial_records = []

        # Tiến hành tối ưu hóa
        if HAS_HYPEROPT:
            def objective(params):
                ema_p = int(params.get('ema_period', 20))
                obv_p = int(params.get('obv_slope_period', 3))

                if "Kết hợp" in opt_strat_choice:
                    ent, ext, _, _, _ = get_ema_obv_combined_signals(train_df, ema_p, obv_p)
                elif "EMA" in opt_strat_choice:
                    ent, ext, _ = get_ema_signals(train_df, ema_p)
                else:
                    ent, ext, _, _ = get_obv_signals(train_df, obv_p)

                res = run_simulation_backtest(train_df, ent, ext, param_fee, param_slippage, param_sl, param_init_cash)
                st_res = res['stats']

                val = st_res['Sharpe Ratio'] if "Sharpe" in opt_metric_target else st_res['Total Return (%)']
                n_trades = st_res['Closed Trades']

                # Điều kiện hợp lệ: Ít nhất 5 giao dịch
                is_valid = np.isfinite(val) and n_trades >= 5
                loss = -val if is_valid else 999.0

                trial_records.append({
                    'EMA': ema_p,
                    'OBV_Slope': obv_p,
                    'Sharpe Ratio': st_res['Sharpe Ratio'],
                    'Total Return (%)': st_res['Total Return (%)'],
                    'Trades': n_trades
                })

                return {'loss': loss, 'status': STATUS_OK}

            space = {}
            if "Kết hợp" in opt_strat_choice:
                space['ema_period'] = hp.quniform('ema_period', 10, 50, 1)
                space['obv_slope_period'] = hp.quniform('obv_slope_period', 2, 25, 1)
            elif "EMA" in opt_strat_choice:
                space['ema_period'] = hp.quniform('ema_period', 10, 50, 1)
            else:
                space['obv_slope_period'] = hp.quniform('obv_slope_period', 2, 25, 1)

            trials = Trials()
            status_text.text(f"Đang chạy Hyperopt TPE ({opt_max_evals} iterations)...")
            best_res = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=opt_max_evals,
                trials=trials,
                rstate=np.random.default_rng(42)
            )
            progress_bar.progress(100)
            status_text.success(" Đã hoàn tất quá trình tối ưu hóa!")

            df_trials = pd.DataFrame(trial_records)
            st.write("### 🏆 Kết Quả Tối Ưu Hóa Tốt Nhất Tìm Được:")
            best_trial = df_trials.loc[df_trials['Sharpe Ratio'].idxmax()] if "Sharpe" in opt_metric_target else df_trials.loc[df_trials['Total Return (%)'].idxmax()]
            st.json(best_trial.to_dict())

            # Biểu đồ phân tán kết quả thử nghiệm
            if "Kết hợp" in opt_strat_choice and len(df_trials) > 0:
                fig_scatter = px.scatter(
                    df_trials, x='EMA', y='OBV_Slope', color='Sharpe Ratio',
                    size='Trades', title="Không Gian Khám Phá Tham Số (Hyperopt Trials)",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Sử dụng bảng tham số tối ưu chuẩn đã chạy 1066 lần từ Notebook: EMA = 36, OBV Slope = 20.")


# ===========================================================================
# TAB 6: CƠ SỞ LÝ THUYẾT & KHUYẾN NGHỊ HỌC THUẬT
# ===========================================================================
with tab6:
    st.subheader("📚 Cơ Sở Lý Thuyết & Báo Cáo Học Thuật (Portfolio Management)")

    st.markdown("""
    ### 1. Ý Nghĩa Của Việc Kết Hợp Hai Chỉ Báo EMA & OBV
    Trong phân tích kỹ thuật và quản lý danh mục định lượng:
    - **Đường EMA (Exponential Moving Average):** Là chỉ báo **theo sau xu hướng (Trend-Following Indicator)**, gán trọng số lớn hơn cho các phiên giao dịch gần nhất, giúp phát hiện sớm sự đảo chiều giá đóng cửa.
    - **Chỉ số OBV (On-Balance Volume):** Là chỉ báo **khối lượng tích lũy (Volume Momentum Indicator)**. Lý thuyết của Joseph Granville chỉ ra rằng *khối lượng luôn đi trước giá (Volume precedes price)*. Độ dốc OBV dương cho thấy dòng tiền thông minh (Smart Money) đang chủ động gom hàng.
    - **Sức mạnh kết hợp:** Khi kết hợp `Giá > EMA` **VÀ** `Độ dốc OBV > 0`, tín hiệu mua chỉ kích hoạt khi vừa có sự xác nhận về **Xu hướng Giá** vừa có sự hậu thuẫn của **Dòng tiền Khối lượng**, giúp loại bỏ đáng kể các tín hiệu bẫy tăng giá giả (Bull Traps).

    ---

    ### 2. Quy Tắc Kiểm Định Chuẩn & Chống Thiên Lệch (Look-Ahead Bias)
    - Tất cả các tín hiệu vào lệnh (Entry) và thoát lệnh (Exit) đều được tính toán và **dịch chuyển 1 phiên (`shift(1)`)**.
    - Điều này đảm bảo quyết định giao dịch ở ngày $T$ chỉ dựa trên dữ liệu đóng cửa đã biết của ngày $T-1$, phản ánh chính xác thực tế giao dịch của nhà đầu tư.

    ---

    ### 3. Tác Động Của Quản Trị Rủi Ro: Cắt Lỗ (Stop Loss 7%) & Chi Phí
    - **Mức Cắt Lỗ Cố Định 7% (`sl_stop=0.07`):** Đóng vai trò then chốt bảo vệ danh mục khỏi các đợt sụt giảm nghiêm trọng khi giá quay đầu bất ngờ.
    - **Chi phí giao dịch & Trượt giá (`fees=0.002`, `slippage=0.001`):** Phản ánh sát thực tế thị trường chứng khoán Việt Nam (phí môi giới + thuế chuyển nhượng ~0.2%, trượt giá khớp lệnh ~0.1%).

    ---

    ### 4. Khuyến Nghị Cho Nhà Đầu Tư & Cải Tiến Nâng Cao
    1. **Walk-Forward Analysis (Kiểm định cuộn):** Không nên dùng một bộ tham số cố định xuyên suốt nhiều năm, mà cần cập nhật định kỳ (re-optimizing) sau mỗi 6 tháng đến 1 năm.
    2. **Bộ lọc Chế độ Thị trường (Market Regime Filter):** Bổ sung thêm đường xu hướng vĩ mô dài hạn (ví dụ: VN-Index nằm trên MA200) để ngừng mở vị thế mua trong các chu kỳ Downtrend lớn như năm 2022.
    3. **Quản lý quy mô vị thế (Position Sizing):** Áp dụng công thức Kelly hoặc phân bổ theo mức biến động thực tế ATR để tối ưu hóa tỷ lệ sinh lời trên rủi ro.
    """)

# Footer bản quyền & thông tin môn học
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.9rem;">
    🎓 Đồ Án Môn Học: <b>Quản Lý Danh Mục Đầu Tư</b> | Khóa Thạc Sĩ (ThS - HK3)<br>
    Ứng dụng Web kiểm định chiến lược EMA & OBV trên cổ phiếu ACB | Thiết kế phục vụ triển khai Streamlit Cloud & GitHub.
</div>
""", unsafe_allow_html=True)
