import streamlit as st
import pandas as pd
import os
import plotly.express as px
import calendar
from datetime import datetime, timedelta
from typing import Tuple, List

# -------------------------- 【修复1：路径适配，本地+云端都能正常找到文件】 --------------------------
# 获取当前app.py文件所在的目录，彻底解决相对路径混乱问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 页面全局配置
st.set_page_config(
    page_title="学生消费行为分析与记账系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- 全局常量（路径已修复） --------------------------
DATA_FILE = os.path.join(DATA_DIR, "expenses.csv")
CATEGORY_FILE = os.path.join(DATA_DIR, "categories.csv")
BUDGET_FILE = os.path.join(DATA_DIR, "budget.csv")

# 餐饮营养标签库
FOOD_TAG_LIST = ["早餐", "正餐", "食堂", "家常菜", "外卖", "快餐", "奶茶", "饮料", "零食", "烧烤", "火锅", "其他餐饮"]


# -------------------------- 工具函数：数据读写 --------------------------
def init_data_files():
    """初始化所有数据文件和文件夹"""
    # 先创建data文件夹，不存在就新建
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    # 初始化分类文件
    if not os.path.exists(CATEGORY_FILE):
        default_categories = ["餐饮", "学习", "娱乐", "交通", "日用品", "其他"]
        pd.DataFrame({"分类名称": default_categories}).to_csv(CATEGORY_FILE, index=False, encoding="utf-8-sig")
    # 初始化预算文件
    if not os.path.exists(BUDGET_FILE):
        pd.DataFrame(columns=["年份", "月份", "预算金额"]).to_csv(BUDGET_FILE, index=False, encoding="utf-8-sig")


@st.cache_data(ttl=60)  # 缓存60秒，避免频繁读文件，同时保证数据更新
def load_categories() -> List[str]:
    """加载消费分类列表"""
    if os.path.exists(CATEGORY_FILE):
        df = pd.read_csv(CATEGORY_FILE, encoding="utf-8-sig")
        return df["分类名称"].tolist()
    return ["餐饮", "学习", "娱乐", "交通", "日用品", "其他"]


def save_categories(categories: List[str]) -> None:
    """保存消费分类列表"""
    pd.DataFrame({"分类名称": categories}).to_csv(CATEGORY_FILE, index=False, encoding="utf-8-sig")


@st.cache_data(ttl=60)
def load_budget(year: int, month: int) -> float:
    """加载指定年月的预算"""
    if os.path.exists(BUDGET_FILE):
        df = pd.read_csv(BUDGET_FILE, encoding="utf-8-sig")
        match = df[(df["年份"] == year) & (df["月份"] == month)]
        if len(match) > 0:
            return float(match["预算金额"].values[0])
    return 0.0


def save_budget(year: int, month: int, amount: float) -> None:
    """保存指定年月的预算"""
    if os.path.exists(BUDGET_FILE):
        df = pd.read_csv(BUDGET_FILE, encoding="utf-8-sig")
    else:
        df = pd.DataFrame(columns=["年份", "月份", "预算金额"])
    mask = (df["年份"] == year) & (df["月份"] == month)
    if len(df[mask]) > 0:
        df.loc[mask, "预算金额"] = round(amount, 2)
    else:
        new_row = pd.DataFrame([{"年份": year, "月份": month, "预算金额": round(amount, 2)}])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(BUDGET_FILE, index=False, encoding="utf-8-sig")


@st.cache_data(ttl=10)  # 数据频繁更新，缓存10秒
def load_data() -> pd.DataFrame:
    """【修复2：统一日期处理，列名统一为“日期”，兜底缺失列】加载消费数据，统一处理格式"""
    # 定义必填的列和对应的数据类型
    REQUIRED_COLUMNS = {
        "日期": "object",
        "金额": "float64",
        "分类": "object",
        "是否餐饮": "bool",
        "备注": "object"
    }

    if os.path.exists(DATA_FILE):
        # 读取csv文件
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

        # ========== 核心修复：检查缺失列，自动补全 ==========
        for col_name, col_type in REQUIRED_COLUMNS.items():
            if col_name not in df.columns:
                # 列不存在，创建空列并设置对应类型
                if col_type == "bool":
                    df[col_name] = False
                elif col_type == "float64":
                    df[col_name] = 0.01
                else:
                    df[col_name] = ""

        # 强制转换日期格式，兜底错误数据
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.date
        # 金额格式处理，兜底无效值
        df["金额"] = pd.to_numeric(df["金额"], errors="coerce").fillna(0.01).round(2)
        # 布尔值处理
        df["是否餐饮"] = df["是否餐饮"].fillna(False).astype(bool)
        # 备注处理
        df["备注"] = df["备注"].fillna("").astype(str)
        # 生成记录序号
        df.insert(0, "记录序号", range(1, len(df) + 1))
        # 过滤掉日期无效的数据
        df = df.dropna(subset=["日期"])
        return df
    else:
        # 空数据时，直接生成带所有必填列的空DataFrame
        return pd.DataFrame(
            columns=["记录序号"] + list(REQUIRED_COLUMNS.keys()),
            dtype=object
        )


def save_data(df: pd.DataFrame) -> None:
    """保存消费数据"""
    save_df = df.drop(columns=["记录序号"])
    save_df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


# -------------------------- 智能分析核心算法 --------------------------
def detect_expense_anomalies(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """消费异常检测，基于分类IQR和日均消费双重判定"""
    if len(df) < 5:
        return pd.DataFrame(), {"异常总笔数": 0, "异常总金额": 0, "单笔异常数": 0, "单日异常数": 0}
    anomaly_records = []
    anomaly_stats = {"单笔异常数": 0, "单日异常数": 0}
    # 1. 单笔消费异常检测（分类IQR法）
    category_group = df.groupby("分类")
    for category, group in category_group:
        if len(group) < 3:
            continue
        q1 = group["金额"].quantile(0.25)
        q3 = group["金额"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        category_anomaly = group[group["金额"] > upper_bound].copy()
        if len(category_anomaly) > 0:
            category_anomaly["异常类型"] = "单笔消费异常"
            category_anomaly["异常原因"] = f"超出[{category}]分类正常消费区间（上限{round(upper_bound, 2)}元）"
            anomaly_records.append(category_anomaly)
            anomaly_stats["单笔异常数"] += len(category_anomaly)
    # 2. 单日消费异常检测（历史日均2倍判定）
    day_group = df.groupby("日期")["金额"].sum().reset_index()
    day_group.columns = ["日期", "当日总消费"]
    if len(day_group) >= 7:
        history_avg = day_group["当日总消费"].mean()
        day_upper_bound = history_avg * 2
        day_anomaly = day_group[day_group["当日总消费"] > day_upper_bound].copy()
        if len(day_anomaly) > 0:
            for _, row in day_anomaly.iterrows():
                day_detail = df[df["日期"] == row["日期"]].copy()
                day_detail["异常类型"] = "单日消费异常"
                day_detail[
                    "异常原因"] = f"当日总消费{round(row['当日总消费'], 2)}元，超出历史日均消费2倍（基准{round(history_avg, 2)}元）"
                anomaly_records.append(day_detail)
            anomaly_stats["单日异常数"] += len(day_anomaly)
    # 合并去重
    if len(anomaly_records) > 0:
        anomaly_df = pd.concat(anomaly_records, ignore_index=True)
        anomaly_df = anomaly_df.drop_duplicates(subset=["记录序号"]).sort_values("日期", ascending=False)
        anomaly_stats["异常总笔数"] = len(anomaly_df)
        anomaly_stats["异常总金额"] = round(anomaly_df["金额"].sum(), 2)
    else:
        anomaly_df = pd.DataFrame()
        anomaly_stats["异常总笔数"] = 0
        anomaly_stats["异常总金额"] = 0.00
    return anomaly_df, anomaly_stats


def analyze_consumption_pattern(df: pd.DataFrame, current_year: int, current_month: int, month_budget: float) -> Tuple[
    dict, dict, str]:
    """消费模式识别与预算预警"""
    pattern_result = {}
    forecast_result = {}
    warning_msg = ""
    if len(df) < 10:
        return pattern_result, forecast_result, "数据量不足，无法完成模式识别与预测，请补充更多消费记录"
    # 消费模式识别
    category_sum = df.groupby("分类")["金额"].sum().sort_values(ascending=False)
    pattern_result["核心消费分类"] = category_sum.head(3).index.tolist()
    pattern_result["分类占比"] = (category_sum / category_sum.sum() * 100).round(2).to_dict()
    # 工作日/周末消费差异
    df_copy = df.copy()
    df_copy["星期"] = pd.to_datetime(df_copy["日期"]).dt.weekday
    df_copy["是否周末"] = df_copy["星期"].apply(lambda x: "周末" if x >= 5 else "工作日")
    weekend_avg = df_copy[df_copy["是否周末"] == "周末"]["金额"].mean()
    weekday_avg = df_copy[df_copy["是否周末"] == "工作日"]["金额"].mean()
    pattern_result["工作日日均消费"] = round(weekday_avg, 2)
    pattern_result["周末日均消费"] = round(weekend_avg, 2)
    pattern_result["消费高峰"] = "周末" if weekend_avg > weekday_avg * 1.5 else "工作日"
    # 当月消费趋势预测
    current_month_df = df[
        (pd.to_datetime(df["日期"]).dt.year == current_year) &
        (pd.to_datetime(df["日期"]).dt.month == current_month)
        ]
    if len(current_month_df) >= 3 and month_budget > 0:
        today = datetime.now().date()
        month_days = calendar.monthrange(current_year, current_month)[1]
        passed_days = min(today.day, month_days)
        used_amount = current_month_df["金额"].sum()
        daily_avg = used_amount / passed_days
        forecast_total = round(daily_avg * month_days, 2)
        forecast_result["当月已过天数"] = passed_days
        forecast_result["当月总天数"] = month_days
        forecast_result["当前日均消费"] = round(daily_avg, 2)
        forecast_result["预测当月总消费"] = forecast_total
        forecast_result["预算金额"] = month_budget
        # 预算预警
        remaining_days = month_days - passed_days
        remaining_budget = month_budget - used_amount
        if remaining_days > 0:
            forecast_result["剩余日均可用额度"] = round(remaining_budget / remaining_days, 2)
        if forecast_total > month_budget:
            over_amount = round(forecast_total - month_budget, 2)
            warning_msg = f"高风险预警：按当前消费速度，预计当月超支{over_amount}元，建议立即控制非必要消费"
        elif used_amount / month_budget >= 0.8:
            warning_msg = f"中度预警：当月预算已使用{round(used_amount / month_budget * 100, 1)}%，剩余额度不足，请合理规划后续消费"
        elif used_amount / month_budget >= 0.5:
            warning_msg = f"温馨提示：当月预算已使用过半，剩余{remaining_days}天，日均可用额度{round(remaining_budget / remaining_days, 2)}元"
        else:
            warning_msg = "消费状态健康：当前消费节奏合理，预算充足，可继续保持"
    return pattern_result, forecast_result, warning_msg


def evaluate_food_health(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame, str]:
    """餐饮健康度评估，基于营养知识库"""
    health_knowledge_base = {
        "早餐": {"score": 10, "label": "健康", "desc": "规律早餐，健康饮食习惯"},
        "正餐": {"score": 10, "label": "健康", "desc": "常规正餐，饮食结构稳定"},
        "食堂": {"score": 9, "label": "健康", "desc": "食堂就餐，饮食相对健康"},
        "家常菜": {"score": 9, "label": "健康", "desc": "家常菜就餐，饮食健康可控"},
        "外卖": {"score": 6, "label": "中等健康", "desc": "外卖高频食用易导致油脂摄入超标"},
        "快餐": {"score": 6, "label": "中等健康", "desc": "快餐多为高热量食品，建议适量食用"},
        "奶茶": {"score": 3, "label": "不健康", "desc": "奶茶含糖量高，高频饮用易导致糖分摄入超标"},
        "饮料": {"score": 4, "label": "不健康", "desc": "含糖饮料过量饮用不利于身体健康"},
        "零食": {"score": 3, "label": "不健康", "desc": "零食多为高油高盐食品，建议减少食用"},
        "烧烤": {"score": 2, "label": "不健康", "desc": "烧烤食品油脂和致癌物含量高，建议少食用"},
        "火锅": {"score": 4, "label": "中等健康", "desc": "火锅饮食易导致油脂和嘌呤摄入超标，建议适量食用"},
        "其他餐饮": {"score": 6, "label": "中等健康", "desc": "未识别到具体餐饮类型，默认中等健康评分"}
    }
    food_df = df[df["是否餐饮"] == True].copy()
    if len(food_df) == 0:
        return {}, pd.DataFrame(), "暂无餐饮消费数据，无法完成健康度评估"

    # 匹配健康评分
    def get_health_score(remark: str) -> Tuple[int, str, str]:
        remark_lower = remark.lower()
        for key, value in health_knowledge_base.items():
            if key in remark_lower:
                return value["score"], value["label"], value["desc"]
        return 6, "中等健康", "未识别到具体餐饮类型，默认中等健康评分"

    food_df[["健康评分", "健康标签", "健康说明"]] = food_df["备注"].apply(
        lambda x: pd.Series(get_health_score(x))
    )
    # 健康度统计
    health_stats = {}
    health_stats["餐饮总笔数"] = len(food_df)
    health_stats["餐饮总金额"] = round(food_df["金额"].sum(), 2)
    health_stats["平均健康评分"] = round(food_df["健康评分"].mean(), 1)
    avg_score = health_stats["平均健康评分"]
    if avg_score >= 8:
        health_stats["健康等级"] = "优秀"
    elif avg_score >= 6:
        health_stats["健康等级"] = "良好"
    elif avg_score >= 4:
        health_stats["健康等级"] = "一般"
    else:
        health_stats["健康等级"] = "较差"
    health_stats["标签占比"] = (food_df["健康标签"].value_counts(normalize=True) * 100).round(2).to_dict()
    # 健康建议
    if health_stats["健康等级"] == "优秀":
        advice = "你的餐饮健康度优秀，饮食结构合理，继续保持当前的健康饮食习惯。"
    elif health_stats["健康等级"] == "良好":
        advice = "你的餐饮健康度良好，整体饮食习惯健康，可适当减少外卖、快餐的食用频次，进一步提升健康度。"
    elif health_stats["健康等级"] == "一般":
        advice = "你的餐饮健康度一般，不健康食品占比偏高，建议减少奶茶、零食、火锅的食用，增加规律正餐的占比。"
    else:
        advice = "警告：你的餐饮健康度较差，高油高糖高盐食品食用过多，建议立即调整饮食结构，减少烧烤、奶茶、零食的食用，养成规律三餐的习惯。"
    # 三餐规律提醒
    food_df["日期"] = pd.to_datetime(food_df["日期"])
    day_food_count = food_df.groupby("日期")["记录序号"].count()
    no_breakfast_days = len(day_food_count[day_food_count < 2])
    if no_breakfast_days > 0:
        advice += f" 另外，你有{no_breakfast_days}天的餐饮消费不足2笔，存在三餐不规律的情况，建议养成规律吃早餐的习惯。"
    return health_stats, food_df, advice


# -------------------------- 页面全局初始化 --------------------------
# 初始化数据文件
init_data_files()
# 加载全局数据
CATEGORY_LIST = load_categories()
df = load_data()
# 时间全局变量
now = datetime.now()
current_year = now.year
current_month = now.month
current_day = now.date()
current_budget = load_budget(current_year, current_month)

# 【修复3：全局统一计算当月消费，避免重复定义，兜底空数据】
if len(df) > 0:
    # 筛选当月消费数据，列名统一用“日期”，彻底解决KeyError
    current_month_expenses = df[
        (pd.to_datetime(df["日期"]).dt.year == current_year) &
        (pd.to_datetime(df["日期"]).dt.month == current_month)
        ]
    used_amount = round(current_month_expenses["金额"].sum(), 2)
else:
    current_month_expenses = pd.DataFrame()  # 空数据时也定义变量，避免未定义报错
    used_amount = 0.00

# 预算剩余计算
remaining_amount = round(current_budget - used_amount, 2) if current_budget > 0 else 0.00

# -------------------------- 侧边栏功能菜单 --------------------------
st.sidebar.title("系统功能菜单")
menu = st.sidebar.radio(
    "请选择功能",
    [
        "首页预算概览",
        "日历记账看板",
        "消费记录管理",
        "记录筛选查询",
        "数据可视化分析",
        "智能消费分析",
        "系统设置"
    ]
)
st.sidebar.divider()
st.sidebar.caption("学生消费行为分析与记账系统")

# -------------------------- 菜单对应页面渲染 --------------------------
# 1. 首页预算概览
if menu == "首页预算概览":
    st.title("首页预算概览")
    st.divider()
    st.subheader(f"{current_year}年{current_month}月 预算总览")

    # 预算总览卡片
    budget_col1, budget_col2, budget_col3 = st.columns(3)
    with budget_col1:
        st.metric("当月总预算", f"{current_budget} 元")
    with budget_col2:
        st.metric("当月已用", f"{used_amount} 元", delta=f"-{used_amount}" if used_amount > 0 else "0")
    with budget_col3:
        delta_color = "normal" if remaining_amount >= 0 else "inverse"
        st.metric("当月剩余", f"{remaining_amount} 元", delta_color=delta_color)

    # 预算进度条
    if current_budget > 0:
        progress_percent = min(used_amount / current_budget, 1.0)
        st.progress(progress_percent, text=f"预算使用进度：{round(progress_percent * 100, 1)}%")
        if progress_percent >= 1.0:
            st.error("警告：本月预算已超支！")
        elif progress_percent >= 0.8:
            st.warning("提示：本月预算已使用80%以上，请注意控制消费。")
        else:
            st.success("当前预算使用状态健康")

    st.divider()
    st.subheader("当月消费统计")

    # 【修复4：兜底空数据，避免无数据时的IndexError】
    if len(current_month_expenses) > 0:
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("消费总笔数", f"{len(current_month_expenses)} 笔")
        with stat_col2:
            day_avg = round(current_month_expenses["金额"].mean(), 2)
            st.metric("单笔平均消费", f"{day_avg} 元")
        with stat_col3:
            # 兜底高频分类，无数据时显示“暂无”
            category_count = current_month_expenses["分类"].value_counts()
            top_category = category_count.index[0] if len(category_count) > 0 else "暂无"
            st.metric("最高频消费分类", top_category)

        st.divider()
        st.subheader("当月最新消费记录")
        latest_df = current_month_expenses.sort_values("日期", ascending=False).head(10)
        st.dataframe(latest_df[["日期", "分类", "金额", "备注"]], use_container_width=True, hide_index=True)
    else:
        st.info("当月暂无消费记录，快去日历记账看板记第一笔账吧！")

# 2. 日历记账看板
elif menu == "日历记账看板":
    st.title("日历记账看板")
    st.divider()
    calendar_col1, calendar_col2 = st.columns([1, 1])

    with calendar_col1:
        selected_date = st.date_input(
            "选择日期",
            value=current_day,
            key="calendar_date"
        )
        st.divider()

        # 当日消费统计
        day_expenses = df[df["日期"] == selected_date]
        day_total = round(day_expenses["金额"].sum(), 2)
        select_year = selected_date.year
        select_month = selected_date.month

        # 当月截至当日统计
        month_to_day_expenses = df[
            (pd.to_datetime(df["日期"]).dt.year == select_year) &
            (pd.to_datetime(df["日期"]).dt.month == select_month) &
            (df["日期"] <= selected_date)
            ]
        month_to_day_total = round(month_to_day_expenses["金额"].sum(), 2)
        select_month_budget = load_budget(select_year, select_month)
        month_days = calendar.monthrange(select_year, select_month)[1]
        day_avg_budget = round(select_month_budget / month_days, 2) if select_month_budget > 0 else 0.00
        day_balance = round(day_avg_budget - day_total, 2) if day_avg_budget > 0 else 0.00

        st.subheader(f"{selected_date} 消费统计")
        stat_col_a, stat_col_b = st.columns(2)
        with stat_col_a:
            st.metric("当日总消费", f"{day_total} 元")
        with stat_col_b:
            if day_avg_budget > 0:
                balance_color = "normal" if day_balance >= 0 else "inverse"
                st.metric("当日预算结余", f"{day_balance} 元", delta_color=balance_color)

        st.divider()
        st.subheader(f"{select_year}年{select_month}月 累计统计（截至{selected_date}）")
        stat_col_c, stat_col_d, stat_col_e = st.columns(3)
        with stat_col_c:
            st.metric("累计总消费", f"{month_to_day_total} 元")
        with stat_col_d:
            if select_month_budget > 0:
                st.metric("当月总预算", f"{select_month_budget} 元")
        with stat_col_e:
            if select_month_budget > 0:
                month_remaining = round(select_month_budget - month_to_day_total, 2)
                month_color = "normal" if month_remaining >= 0 else "inverse"
                st.metric("当月剩余预算", f"{month_remaining} 元", delta_color=month_color)

    with calendar_col2:
        st.subheader(f"{selected_date} 消费明细")
        if len(day_expenses) > 0:
            show_df = day_expenses[["分类", "金额", "是否餐饮", "备注"]].reset_index(drop=True)
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        else:
            st.info("该日期暂无消费记录")

        st.divider()
        st.subheader("快捷新增当日消费")
        with st.form("quick_add_form", border=True):
            quick_col1, quick_col2 = st.columns(2)
            final_quick_remark = ""
            with quick_col1:
                quick_amount = st.number_input("消费金额（元）", min_value=0.01, step=0.01, format="%.2f",
                                               key="quick_amount")
                quick_category = st.selectbox("消费分类", options=CATEGORY_LIST, key="quick_category")
            with quick_col2:
                quick_is_food = st.checkbox("是否为餐饮消费", value=(quick_category == "餐饮"), key="quick_food")
                if quick_is_food:
                    quick_food_tag = st.selectbox("餐饮类型标签（用于健康度评估）", options=FOOD_TAG_LIST,
                                                  key="quick_food_tag")
                    quick_remark_supplement = st.text_input("备注补充（选填）", placeholder="比如：黄焖鸡米饭、可乐",
                                                            key="quick_remark_supplement")
                    final_quick_remark = quick_food_tag + (
                        " " + quick_remark_supplement if quick_remark_supplement else "")
                else:
                    final_quick_remark = st.text_input("备注（选填）", placeholder="比如：买教材、公交费",
                                                       key="quick_remark_normal")
            quick_submit = st.form_submit_button("保存消费记录", use_container_width=True)

        if quick_submit:
            new_data = pd.DataFrame([{
                "日期": str(selected_date),
                "金额": round(quick_amount, 2),
                "分类": quick_category,
                "是否餐饮": quick_is_food,
                "备注": final_quick_remark
            }])
            if os.path.exists(DATA_FILE):
                new_data.to_csv(DATA_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
            else:
                new_data.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
            st.success("消费记录保存成功！")
            st.rerun()

# 3. 消费记录管理
elif menu == "消费记录管理":
    st.title("消费记录管理")
    st.divider()
    if len(df) == 0:
        st.info("暂无消费记录，无需进行管理操作")
    else:
        tab1, tab2, tab3 = st.tabs(["编辑已有记录", "单条删除记录", "批量删除记录"])
        with tab1:
            edit_id = st.selectbox(
                "选择要编辑的记录序号",
                options=df["记录序号"].tolist(),
                format_func=lambda
                    x: f"序号{x} | {df.loc[df['记录序号'] == x, '日期'].values[0]} | {df.loc[df['记录序号'] == x, '分类'].values[0]} | {df.loc[df['记录序号'] == x, '金额'].values[0]}元",
                key="edit_select"
            )
            edit_row = df[df["记录序号"] == edit_id].iloc[0].copy()
            category_index = CATEGORY_LIST.index(edit_row["分类"]) if edit_row["分类"] in CATEGORY_LIST else 0
            final_edit_remark = edit_row["备注"]

            with st.form("expense_edit_form", border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    edit_date = st.date_input("消费日期", value=edit_row["日期"], key="edit_date")
                    edit_amount = st.number_input("消费金额（元）", min_value=0.01, step=0.01, format="%.2f",
                                                  value=float(edit_row["金额"]), key="edit_amount")
                with col2:
                    edit_category = st.selectbox("消费分类", options=CATEGORY_LIST, index=category_index,
                                                 key="edit_category")
                    edit_is_food = st.checkbox("是否为餐饮消费", value=bool(edit_row["是否餐饮"]), key="edit_food")
                with col3:
                    if edit_is_food:
                        origin_tag = "其他餐饮"
                        for tag in FOOD_TAG_LIST:
                            if tag in edit_row["备注"]:
                                origin_tag = tag
                                break
                        edit_food_tag = st.selectbox("餐饮类型标签（用于健康度评估）", options=FOOD_TAG_LIST,
                                                     index=FOOD_TAG_LIST.index(origin_tag), key="edit_food_tag")
                        edit_remark_supplement = st.text_input("备注补充（选填）",
                                                               value=edit_row["备注"].replace(origin_tag, "").strip(),
                                                               key="edit_remark_supplement")
                        final_edit_remark = edit_food_tag + (
                            " " + edit_remark_supplement if edit_remark_supplement else "")
                    else:
                        final_edit_remark = st.text_input("备注（选填）", value=edit_row["备注"],
                                                          key="edit_remark_normal")
                edit_submitted = st.form_submit_button("确认修改记录", use_container_width=True)

            if edit_submitted:
                df.loc[df["记录序号"] == edit_id, "日期"] = str(edit_date)
                df.loc[df["记录序号"] == edit_id, "金额"] = round(edit_amount, 2)
                df.loc[df["记录序号"] == edit_id, "分类"] = edit_category
                df.loc[df["记录序号"] == edit_id, "是否餐饮"] = edit_is_food
                df.loc[df["记录序号"] == edit_id, "备注"] = final_edit_remark
                save_data(df)
                st.success("记录修改成功！")
                st.rerun()

        with tab2:
            delete_id = st.selectbox(
                "选择要删除的记录序号",
                options=df["记录序号"].tolist(),
                format_func=lambda
                    x: f"序号{x} | {df.loc[df['记录序号'] == x, '日期'].values[0]} | {df.loc[df['记录序号'] == x, '分类'].values[0]} | {df.loc[df['记录序号'] == x, '金额'].values[0]}元",
                key="single_delete"
            )
            st.write("选中的记录详情：")
            st.dataframe(df[df["记录序号"] == delete_id], use_container_width=True, hide_index=True)
            delete_btn = st.button("确认删除该记录", use_container_width=True, type="primary")
            if delete_btn:
                new_df = df[df["记录序号"] != delete_id]
                save_data(new_df)
                st.success("记录删除成功！")
                st.rerun()

        with tab3:
            delete_ids = st.multiselect(
                "选择要批量删除的记录序号",
                options=df["记录序号"].tolist(),
                format_func=lambda
                    x: f"序号{x} | {df.loc[df['记录序号'] == x, '日期'].values[0]} | {df.loc[df['记录序号'] == x, '分类'].values[0]} | {df.loc[df['记录序号'] == x, '金额'].values[0]}元",
                key="batch_delete"
            )
            if len(delete_ids) > 0:
                st.write("已选中的记录列表：")
                st.dataframe(df[df["记录序号"].isin(delete_ids)], use_container_width=True, hide_index=True)
                batch_delete_btn = st.button("确认批量删除选中记录", use_container_width=True, type="primary")
                if batch_delete_btn:
                    new_df = df[~df["记录序号"].isin(delete_ids)]
                    save_data(new_df)
                    st.success("批量删除成功！")
                    st.rerun()
            else:
                st.info("请至少选择一条要删除的记录")

# 4. 记录筛选查询
elif menu == "记录筛选查询":
    st.title("消费记录筛选查询")
    st.divider()
    if len(df) == 0:
        st.info("暂无消费记录")
    else:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            date_min = df["日期"].min()
            date_max = df["日期"].max()
            date_range: Tuple[datetime.date, datetime.date] = st.date_input(
                "选择日期范围",
                value=(date_min, date_max),
                min_value=date_min,
                max_value=date_max,
                key="filter_date"
            )
        with filter_col2:
            category_unique = sorted(df["分类"].unique().tolist())
            category_options = ["全部"] + category_unique
            selected_category = st.selectbox("选择消费分类", options=category_options, key="filter_category")

        filter_df = df.copy()
        if len(date_range) == 2:
            start_date, end_date = date_range
            filter_df = filter_df[(filter_df["日期"] >= start_date) & (filter_df["日期"] <= end_date)]
        if selected_category != "全部":
            filter_df = filter_df[filter_df["分类"] == selected_category]

        st.subheader("筛选结果统计")
        total_money = round(filter_df["金额"].sum(), 2)
        avg_money = round(filter_df["金额"].mean(), 2) if len(filter_df) > 0 else 0.00
        record_count = len(filter_df)
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("筛选范围内总消费", f"{total_money} 元")
        with stat_col2:
            st.metric("筛选范围内单笔平均消费", f"{avg_money} 元")
        with stat_col3:
            st.metric("筛选范围内总记录数", f"{record_count} 条")

        st.divider()
        st.subheader("筛选结果详情")
        st.dataframe(filter_df, use_container_width=True, hide_index=True)

# 5. 数据可视化分析
elif menu == "数据可视化分析":
    st.title("消费数据可视化分析")
    st.divider()
    if len(df) == 0:
        st.info("暂无消费数据，无法生成分析图表")
    else:
        # 时间维度切换器
        st.subheader("时间维度选择")
        time_dimension = st.radio("选择统计周期", ["每日", "每月", "每年"], horizontal=True)
        st.divider()

        # 数据预处理
        df_viz = df.copy()
        df_viz["日期"] = pd.to_datetime(df_viz["日期"])
        if time_dimension == "每日":
            df_viz["时间维度"] = df_viz["日期"].dt.date
            x_title = "日期"
            chart_title = "每日消费金额趋势（分分类堆叠）"
            line_title = "每日总消费金额变化趋势"
        elif time_dimension == "每月":
            df_viz["时间维度"] = df_viz["日期"].dt.to_period("M").astype(str)
            x_title = "月份"
            chart_title = "每月消费金额趋势（分分类堆叠）"
            line_title = "每月总消费金额变化趋势"
        else:  # 每年
            df_viz["时间维度"] = df_viz["日期"].dt.year.astype(str)
            x_title = "年份"
            chart_title = "每年消费金额趋势（分分类堆叠）"
            line_title = "每年总消费金额变化趋势"

        # 数据聚合
        dimension_category_sum = df_viz.groupby(["时间维度", "分类"], as_index=False)["金额"].sum()
        dimension_total_sum = df_viz.groupby("时间维度", as_index=False)["金额"].sum()

        # 图表1：消费分类总占比饼图
        st.subheader("全周期消费分类金额总占比")
        category_total_sum = df_viz.groupby("分类", as_index=False)["金额"].sum()
        pie_fig = px.pie(
            category_total_sum,
            values="金额",
            names="分类",
            title="全周期各分类消费金额总占比",
            hole=0.3
        )
        pie_fig.update_traces(textposition="inside", texttemplate="%{value:.2f}")
        pie_fig.update_layout(height=500)
        st.plotly_chart(pie_fig, use_container_width=True)

        st.divider()
        # 图表2：分分类堆叠柱状图
        st.subheader(chart_title)
        bar_fig = px.bar(
            dimension_category_sum,
            x="时间维度",
            y="金额",
            color="分类",
            title=chart_title,
            barmode="stack"
        )
        bar_fig.update_layout(
            height=500,
            xaxis_title=x_title,
            yaxis_title="消费金额（元）",
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        st.divider()
        # 图表3：总消费趋势折线图
        st.subheader(line_title)
        line_fig = px.line(
            dimension_total_sum,
            x="时间维度",
            y="金额",
            title=line_title,
            markers=True
        )
        line_fig.update_layout(
            height=500,
            xaxis_title=x_title,
            yaxis_title="消费金额（元）"
        )
        st.plotly_chart(line_fig, use_container_width=True)

# 6. 智能消费分析
elif menu == "智能消费分析":
    st.title("智能消费分析")
    st.divider()
    if len(df) < 10:
        st.info("消费记录数据量不足，需至少10条记录才能启动智能分析，请补充更多消费数据")
    else:
        ai_tab1, ai_tab2, ai_tab3 = st.tabs([
            "消费异常智能检测",
            "消费模式识别与预算预警",
            "餐饮健康度评估"
        ])
        # 消费异常智能检测
        with ai_tab1:
            st.subheader("消费异常智能检测结果")
            st.caption("判定规则：1.单笔消费超出同分类历史正常区间（IQR法）；2.单日总消费超出历史日均消费2倍")
            st.divider()

            anomaly_df, anomaly_stats = detect_expense_anomalies(df)
            anomaly_col1, anomaly_col2, anomaly_col3, anomaly_col4 = st.columns(4)
            with anomaly_col1:
                st.metric("异常消费总笔数", anomaly_stats["异常总笔数"])
            with anomaly_col2:
                st.metric("异常消费总金额", f"{anomaly_stats['异常总金额']} 元")
            with anomaly_col3:
                st.metric("单笔异常消费数", anomaly_stats["单笔异常数"])
            with anomaly_col4:
                st.metric("单日异常消费数", anomaly_stats["单日异常数"])

            st.divider()
            if len(anomaly_df) > 0:
                st.subheader("异常消费记录明细")
                show_anomaly_df = anomaly_df[["记录序号", "日期", "分类", "金额", "异常类型", "异常原因", "备注"]]
                st.dataframe(show_anomaly_df, use_container_width=True, hide_index=True)
            else:
                st.success("恭喜！未检测到异常消费记录，你的消费行为稳定合理")

        # 消费模式识别与预算预警
        with ai_tab2:
            st.subheader("消费模式识别与趋势预测")
            pattern_result, forecast_result, warning_msg = analyze_consumption_pattern(df, current_year, current_month,
                                                                                       current_budget)

            # 预警提示
            if "高风险预警" in warning_msg:
                st.error(warning_msg)
            elif "中度预警" in warning_msg:
                st.warning(warning_msg)
            else:
                st.success(warning_msg)

            st.divider()
            st.subheader("你的消费模式识别结果")
            if pattern_result:
                pattern_col1, pattern_col2 = st.columns(2)
                with pattern_col1:
                    st.write("核心消费分类TOP3")
                    for i, category in enumerate(pattern_result["核心消费分类"], 1):
                        st.write(f"{i}. {category}（占比{pattern_result['分类占比'][category]}%）")
                    st.divider()
                    st.write(f"消费高峰时段：{pattern_result['消费高峰']}")
                with pattern_col2:
                    st.metric("工作日日均消费", f"{pattern_result['工作日日均消费']} 元")
                    st.metric("周末日均消费", f"{pattern_result['周末日均消费']} 元")
            else:
                st.info("数据量不足，无法完成消费模式识别")

            st.divider()
            st.subheader("当月消费趋势预测与预算规划")
            if forecast_result:
                forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
                with forecast_col1:
                    st.metric("当月已过天数", f"{forecast_result['当月已过天数']}/{forecast_result['当月总天数']}天")
                    st.metric("当前日均消费", f"{forecast_result['当前日均消费']} 元")
                with forecast_col2:
                    st.metric("预测当月总消费", f"{forecast_result['预测当月总消费']} 元")
                    st.metric("当月预算金额", f"{forecast_result['预算金额']} 元")
                with forecast_col3:
                    if "剩余日均可用额度" in forecast_result:
                        st.metric("剩余日均可用额度", f"{forecast_result['剩余日均可用额度']} 元")
                st.caption("预测说明：基于当月已消费数据线性拟合，预测全月消费情况，为预算规划提供数据支撑")
            else:
                st.info("未设置当月预算或当月数据不足，无法完成趋势预测")

        # 餐饮健康度评估
        with ai_tab3:
            st.subheader("餐饮消费健康度评估")
            health_stats, food_detail_df, health_advice = evaluate_food_health(df)

            if health_stats:
                health_level = health_stats["健康等级"]
                if health_level == "优秀":
                    st.success(f"你的餐饮健康度等级：{health_level}，平均健康评分{health_stats['平均健康评分']}/10")
                elif health_level == "良好":
                    st.info(f"你的餐饮健康度等级：{health_level}，平均健康评分{health_stats['平均健康评分']}/10")
                elif health_level == "一般":
                    st.warning(f"你的餐饮健康度等级：{health_level}，平均健康评分{health_stats['平均健康评分']}/10")
                else:
                    st.error(f"你的餐饮健康度等级：{health_level}，平均健康评分{health_stats['平均健康评分']}/10")

                st.write(health_advice)
                st.divider()

                health_col1, health_col2, health_col3 = st.columns(3)
                with health_col1:
                    st.metric("餐饮消费总笔数", health_stats["餐饮总笔数"])
                with health_col2:
                    st.metric("餐饮消费总金额", f"{health_stats['餐饮总金额']} 元")
                with health_col3:
                    st.metric("平均健康评分", f"{health_stats['平均健康评分']}/10")

                st.divider()
                st.subheader("餐饮健康标签占比")
                label_df = pd.DataFrame({
                    "健康标签": list(health_stats["标签占比"].keys()),
                    "占比(%)": list(health_stats["标签占比"].values())
                })
                health_pie = px.pie(
                    label_df,
                    values="占比(%)",
                    names="健康标签",
                    title="餐饮健康标签分布",
                    color_discrete_map={"健康": "green", "中等健康": "orange", "不健康": "red"}
                )
                st.plotly_chart(health_pie, use_container_width=True)

                st.divider()
                st.subheader("餐饮消费健康明细")
                show_food_df = food_detail_df[["记录序号", "日期", "金额", "备注", "健康评分", "健康标签", "健康说明"]]
                st.dataframe(show_food_df, use_container_width=True, hide_index=True)
            else:
                st.info(health_advice)

# 7. 系统设置
elif menu == "系统设置":
    st.title("系统设置")
    st.divider()
    setting_tab1, setting_tab2 = st.tabs(["自定义消费分类", "每月预算设置"])

    with setting_tab1:
        st.subheader("管理消费分类")
        current_cats = load_categories()
        st.write("当前消费分类列表：")
        for i, cat in enumerate(current_cats, 1):
            st.write(f"{i}. {cat}")

        st.divider()
        st.subheader("添加新分类")
        new_cat = st.text_input("输入新的消费分类名称", placeholder="比如：医疗、通讯", key="new_cat")
        add_cat_btn = st.button("添加分类", use_container_width=True, key="add_cat")
        if add_cat_btn:
            if new_cat and new_cat not in current_cats:
                current_cats.append(new_cat)
                save_categories(current_cats)
                st.success(f"分类「{new_cat}」添加成功！")
                st.rerun()
            elif new_cat in current_cats:
                st.error("该分类已存在，请勿重复添加")
            else:
                st.error("请输入有效的分类名称")

        st.divider()
        st.subheader("删除分类")
        deletable_cats = [cat for cat in current_cats if cat not in ["餐饮", "学习", "娱乐", "交通", "日用品", "其他"]]
        if len(deletable_cats) > 0:
            delete_cat = st.selectbox("选择要删除的分类（默认分类不可删除）", options=deletable_cats, key="delete_cat")
            delete_cat_btn = st.button("删除分类", use_container_width=True, type="primary", key="delete_cat_btn")
            if delete_cat_btn:
                current_cats.remove(delete_cat)
                save_categories(current_cats)
                st.success(f"分类「{delete_cat}」删除成功！")
                st.rerun()
        else:
            st.info("暂无可删除的自定义分类（默认分类不可删除）")

    with setting_tab2:
        st.subheader("设置每月预算")
        col_year, col_month = st.columns(2)
        with col_year:
            set_year = st.number_input("选择年份", min_value=2020, max_value=2030, value=current_year, step=1,
                                       key="set_year")
        with col_month:
            set_month = st.number_input("选择月份", min_value=1, max_value=12, value=current_month, step=1,
                                        key="set_month")
        existing_budget = load_budget(int(set_year), int(set_month))
        set_amount = st.number_input(
            f"设置{int(set_year)}年{int(set_month)}月的预算金额（元）",
            min_value=0.01,
            step=100.00,
            format="%.2f",
            value=existing_budget if existing_budget > 0 else 1000.00,
            key="set_amount"
        )
        save_budget_btn = st.button("保存预算设置", use_container_width=True, key="save_budget")
        if save_budget_btn:
            save_budget(int(set_year), int(set_month), set_amount)
            st.success(f"{int(set_year)}年{int(set_month)}月的预算已设置为 {set_amount} 元！")
            st.rerun()