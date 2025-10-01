import time
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="AutoInit Linear Regression (CRISP-DM)", page_icon="📈", layout="wide")

# ---------- Sidebar: CRISP-DM ----------
with st.sidebar:
    st.title("CRISP-DM 導覽")
    steps = [
        ("Business Understanding", "用線性回歸擬合 y ~ x"),
        ("Data Understanding", "檢視資料分佈與噪聲"),
        ("Data Preparation", "切分 train/test"),
        ("Modeling", "亂數初始值 + 梯度下降（含穩定化）/ sklearn 對照"),
        ("Evaluation", "R²、RMSE、MAE、Loss Curve"),
        ("Deployment", "本機 & Streamlit Cloud")
    ]
    for name, desc in steps:
        st.markdown(f"- **{name}** — {desc}")

# ---------- Header ----------
st.title("📈 線性回歸（自動亂數初始值 + 視覺化）— sklearn + CRISP-DM")
st.caption("內建數值穩定化：特徵標準化、學習率自動回退（loss 上升時），避免 NaN/Inf。")

# ---------- Controls ----------
colA, colB, colC, colD = st.columns([1.2, 1.2, 1.2, 1.4], vertical_alignment="bottom")
with colA:
    n_samples = st.slider("樣本數", 30, 1000, 200, step=10)
with colB:
    noise = st.slider("噪聲標準差", 0.0, 10.0, 2.5, 0.1)
with colC:
    lr_init = st.slider("初始學習率 (GD)", 1e-5, 1e-1, 1e-2, format="%.5f")
with colD:
    n_iters = st.slider("最大迭代次數", 10, 3000, 500, step=10)

colE, colF, colG = st.columns([1.2, 1.2, 1.6], vertical_alignment="bottom")
with colE:
    seed_data = st.number_input("資料亂數種子", min_value=0, value=int(time.time()) % 10**6, step=1)
with colF:
    seed_init = st.number_input("參數初始亂數種子", min_value=0, value=(int(time.time()*1.7) % 10**6), step=1)
with colG:
    use_standardize = st.checkbox("使用特徵標準化（建議）", value=True, help="對 X、y 做標準化再用 GD，最後再轉回原始座標以便繪圖/評估。")

# ---------- Data Generation ----------
rng_data = np.random.default_rng(seed_data)
true_w = rng_data.normal(loc=3.0, scale=1.0)     # 真實斜率
true_b = rng_data.normal(loc=1.0, scale=2.0)     # 真實截距
X = rng_data.uniform(-10, 10, size=(n_samples, 1))
y = true_w * X[:, 0] + true_b + rng_data.normal(0, noise, size=n_samples)

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- (可選) 標準化：穩定 GD ----------
def standardize(train_arr, test_arr):
    mean = np.mean(train_arr)
    std = np.std(train_arr)
    std = std if std > 1e-12 else 1.0  # 避免除以 0
    return (train_arr - mean) / std, (test_arr - mean) / std, mean, std

if use_standardize:
    Xtr_s, Xte_s, mx, sx = standardize(X_train[:, 0], X_test[:, 0])
    ytr_s, yte_s, my, sy = standardize(y_train, y_test)
    Xtr_s = Xtr_s.reshape(-1, 1)
    Xte_s = Xte_s.reshape(-1, 1)
else:
    # 保持原尺度
    Xtr_s, Xte_s = X_train.copy(), X_test.copy()
    ytr_s, yte_s = y_train.copy(), y_test.copy()
    mx = np.mean(X_train[:, 0]); sx = np.std(X_train[:, 0]); sx = sx if sx > 1e-12 else 1.0
    my = np.mean(y_train);       sy = np.std(y_train);       sy = sy if sy > 1e-12 else 1.0

# ---------- Random Init for Gradient Descent ----------
rng_init = np.random.default_rng(seed_init)
w_s = rng_init.normal(loc=0.0, scale=1.0)  # 在標準化空間初始化
b_s = rng_init.normal(loc=0.0, scale=1.0)

# ---------- Gradient Descent with safety ----------
def finite(x): 
    return np.all(np.isfinite(x))

loss_hist = []
wb_hist = []
lr = lr_init
best_loss = np.inf
best_wb = (w_s, b_s)
patience = 30  # 容許若干次上升後仍持續嘗試
bad_steps = 0

for t in range(n_iters):
    # 預測 (在標準化空間)
    y_pred_s = w_s * Xtr_s[:, 0] + b_s
    err = y_pred_s - ytr_s
    loss = float(np.mean(err ** 2))

    # NaN/Inf 立即處理
    if not finite([loss]):
        bad_steps += 1
        lr *= 0.5
        if lr < 1e-7:
            st.warning("偵測到數值不穩定，已提前停止。建議降低學習率或勾選標準化。")
            break
        continue

    loss_hist.append(loss)
    wb_hist.append((w_s, b_s))

    # 紀錄最佳
    if loss < best_loss:
        best_loss = loss
        best_wb = (w_s, b_s)
        bad_steps = 0
    else:
        bad_steps += 1
        # 若 loss 上升，回退一步並降低 lr
        w_s, b_s = best_wb
        lr *= 0.5

    # 梯度
    dw = 2.0 * np.mean(err * Xtr_s[:, 0])
    db = 2.0 * np.mean(err)

    # 更新前檢查
    w_new = w_s - lr * dw
    b_new = b_s - lr * db

    if not finite([w_new, b_new]):
        lr *= 0.5
        if lr < 1e-7:
            st.warning("偵測到數值不穩定，已提前停止。建議降低學習率或勾選標準化。")
            break
        continue

    w_s, b_s = w_new, b_new

    # 若不斷變差也停止
    if bad_steps > patience:
        st.info("多次 loss 上升，已提前停止並採用最佳參數。可降低學習率或增加標準化。")
        break

# 使用最佳參數
w_s, b_s = best_wb

# ---------- 轉回原始空間係數（便於繪圖與與 sklearn 對照） ----------
# 標準化定義：Xs = (x - mx) / sx，ys = (y - my) / sy
# 於標準化空間：ys ≈ w_s * Xs + b_s
# => y ≈ sy * (w_s * (x - mx)/sx + b_s) + my
# => 斜率 w = sy * w_s / sx
#    截距 b = sy * (b_s - w_s * mx / sx) + my
gd_w = float(sy * w_s / sx)
gd_b = float(sy * (b_s - w_s * mx / sx) + my)

# ---------- sklearn LinearRegression 對照 ----------
sk_model = LinearRegression(fit_intercept=True)
sk_model.fit(X_train, y_train)
sk_w = float(sk_model.coef_[0])
sk_b = float(sk_model.intercept_)

# ---------- Evaluation ----------
def eval_metrics(w_, b_, X_, y_):
    pred = w_ * X_[:, 0] + b_
    # 安全保護：若出現非有限數，回傳 NaN-friendly 結果
    if not np.all(np.isfinite(pred)):
        return np.nan, np.nan, np.nan
    r2 = r2_score(y_, pred)
    rmse = float(np.sqrt(mean_squared_error(y_, pred)))
    mae = float(mean_absolute_error(y_, pred))
    return r2, rmse, mae

r2_train_gd, rmse_train_gd, mae_train_gd = eval_metrics(gd_w, gd_b, X_train, y_train)
r2_test_gd,  rmse_test_gd,  mae_test_gd  = eval_metrics(gd_w, gd_b, X_test,  y_test)
r2_train_sk, rmse_train_sk, mae_train_sk = eval_metrics(sk_w, sk_b, X_train, y_train)
r2_test_sk,  rmse_test_sk,  mae_test_sk  = eval_metrics(sk_w, sk_b, X_test,  y_test)

# ---------- Plots ----------
left, right = st.columns([1.1, 1])

with left:
    st.subheader("資料散點圖與模型擬合")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(X[:, 0], y, alpha=0.45, label="資料點")
    xs = np.linspace(X.min(), X.max(), 200)
    ax.plot(xs, true_w * xs + true_b, linestyle="--", label="真實函數")
    ax.plot(xs, gd_w * xs + gd_b, label="GD 擬合線（亂數初始+穩定化）")
    ax.plot(xs, sk_w * xs + sk_b, label="sklearn 擬合線")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Loss Curve（MSE）")
    fig2, ax2 = plt.subplots(figsize=(6.8, 3.2))
    if len(loss_hist) > 0 and np.all(np.isfinite(loss_hist)):
        ax2.plot(loss_hist)
    else:
        ax2.plot([])
        ax2.text(0.5, 0.5, "（已啟動數值保護，未記錄有效 Loss）", ha="center", va="center")
    ax2.set_xlabel("迭代次數")
    ax2.set_ylabel("MSE")
    st.pyplot(fig2, use_container_width=True)

with right:
    st.subheader("參數與評估")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("GD 斜率 w", f"{gd_w:.6f}")
        st.metric("GD 截距 b", f"{gd_b:.6f}")
        st.caption("（標準化 + 自動回退學習率 + 安全防護）")
    with m2:
        st.metric("sklearn 斜率 w", f"{sk_w:.6f}")
        st.metric("sklearn 截距 b", f"{sk_b:.6f}")

    st.markdown("### 驗證集（Test）表現")
    def fmt(x): 
        return "NaN" if (x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))) else f"{x:.6f}"
    st.write(
        f"- **GD**：R² = {fmt(r2_test_gd)}｜RMSE = {fmt(rmse_test_gd)}｜MAE = {fmt(mae_test_gd)}\n"
        f"- **sklearn**：R² = {fmt(r2_test_sk)}｜RMSE = {fmt(rmse_test_sk)}｜MAE = {fmt(mae_test_sk)}"
    )

st.divider()
st.markdown(
    "### CRISP-DM 小結\n"
    "- **Business Understanding**：以線性回歸預測 y ~ x。\n"
    "- **Data Understanding**：可視化散點與噪聲。\n"
    "- **Data Preparation**：train/test split（80/20），（可選）標準化。\n"
    "- **Modeling**：亂數初始 + 梯度下降，含學習率回退與數值保護；對照 sklearn。\n"
    "- **Evaluation**：R² / RMSE / MAE 與 Loss Curve。\n"
    "- **Deployment**：本機 `streamlit run app.py`，或部署到 Streamlit Cloud。"
)

