import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# تنظیم فونت فارسی در matplotlib (در این مثال از فونت "Vazir" استفاده شده است)
plt.rcParams['font.family'] = 'Parnian'
plt.rcParams['font.size'] = 10

st.title("بهینه‌سازی سرمایه‌گذاری در استان هرمزگان برای رسیدن به رشد اقتصادی 6 درصد (بدون محدودیت سقف بودجه)")

# تنظیمات ورودی در نوار کناری (sidebar)
st.sidebar.header("تنظیمات ورودی")
target_growth_percent = st.sidebar.number_input(
    "انتخاب رشد اسمی(رشد واقعی بعلاوه تورم) هدف (%)", value=41.0, min_value=0.0, max_value=200.0, step=1.0
)
epsilon_percent = st.sidebar.number_input(
    "انتخاب تورم مورد انتظار (%)", value=35.0, min_value=0.0, max_value=100.0, step=1.0
)
epsilon = epsilon_percent / 100.0  # تبدیل درصد به کسر

# آپلود فایل اکسل ورودی
uploaded_file = st.file_uploader("آپلود فایل اکسل داده‌ها (فرمت xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # خواندن فایل اکسل
        xls = pd.ExcelFile(uploaded_file)
    except Exception as e:
        st.error("خطا در بارگذاری فایل اکسل: " + str(e))
    else:
        try:
            # بارگذاری ماتریس‌های ورودی از برگه‌های مشخص
            L_inv_df = pd.read_excel(xls, sheet_name="Leontief_Inverse", header=None, skiprows=1)
            L_inv_df = L_inv_df.iloc[:, 1:]
            L_inv = L_inv_df.to_numpy()
            
            A_df = pd.read_excel(xls, sheet_name="coefficient_matrix", header=None, skiprows=1)
            A_df = A_df.iloc[:, 1:]
            A = A_df.to_numpy()
            
            W_df = pd.read_excel(xls, sheet_name="Gross Value Added (W)", header=None, skiprows=1)
            W_df = W_df.iloc[:, 1:]
            g = W_df.to_numpy().flatten()  # خروجی پایه (ارزش افزوده)
            
            n = len(g)  # تعداد بخش‌ها
        except Exception as e:
            st.error("خطا در خواندن برگه‌های اکسل: " + str(e))
            st.stop()

        # محاسبه شاخص‌های لینکج (Backward & Forward)
        BL = L_inv.sum(axis=1)
        I = np.eye(n)
        try:
            G_inv = np.linalg.inv(I - A.T)
        except Exception as exc:
            st.error("خطا در محاسبه معکوس گوهوش (Ghosh Inverse): " + str(exc))
            st.stop()
        FL = G_inv.sum(axis=0)
        
        # نرمال‌سازی شاخص‌های لینکج
        BL_norm = (BL - BL.min()) / (BL.max() - BL.min())
        FL_norm = (FL - FL.min()) / (FL.max() - FL.min())
        
        # محاسبه ضریب تغییرات برای ماتریس A
        mean_rows = np.mean(A, axis=1)
        mean_cols = np.mean(A, axis=0)
        mean_rows[mean_rows == 0] = np.nan
        mean_cols[mean_cols == 0] = np.nan
        cv_BL = np.std(A, axis=1) / mean_rows
        cv_FL = np.std(A, axis=0) / mean_cols
        cv_BL_norm = (cv_BL - np.nanmin(cv_BL)) / (np.nanmax(cv_BL) - np.nanmin(cv_BL))
        cv_FL_norm = (cv_FL - np.nanmin(cv_FL)) / (np.nanmax(cv_FL) - np.nanmin(cv_FL))
        
        # تعیین وزن هزینه سرمایه‌گذاری برای هر بخش
        lambda1 = 0.5
        lambda2 = 0.5
        mu1 = 0.0
        mu2 = 0.0
        w_cost = 1 + lambda1 * (1 - BL_norm) + lambda2 * (1 - FL_norm) + mu1 * cv_BL_norm + mu2 * cv_FL_norm
        
        # تعیین رشد کل هدف: به عنوان درصدی از ارزش افزوده کل
        target_growth_total = (target_growth_percent / 100.0) * np.sum(g)
        
        st.write("تعداد بخش‌ها:", n)
        
        # تنظیم محدودیت‌های بهینه‌سازی خطی:
        # 1. محدودیت رشد هر بخش: (L_inv @ x)_i >= epsilon * g[i]
        A_individual = -L_inv
        b_individual = -epsilon * g
        
        # 2. محدودیت رشد کل: sum(L_inv @ x) >= target_growth_total
        A_agg = -np.sum(L_inv, axis=0).reshape(1, -1)
        b_agg = -target_growth_total
        
        # ترکیب محدودیت‌ها (بدون محدودیت سقف بودجه)
        A_ub = np.vstack([A_agg, A_individual])
        b_ub = np.concatenate(([b_agg], b_individual))
        
        # تابع هدف: کمینه کردن مجموع سرمایه‌گذاری‌های وزن‌دار
        c = w_cost
        bounds = [(0, None) for _ in range(n)]
        
        st.write("در حال اجرای مدل بهینه‌سازی خطی...")
        res_lin = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res_lin.success:
            x_opt_lin = res_lin.x
            total_budget_lin = np.sum(x_opt_lin)
            induced_growth_lin = L_inv @ x_opt_lin
            rel_growth_lin = induced_growth_lin / g  # رشد نسبی هر بخش
            
            st.success("مدل بهینه‌سازی خطی با موفقیت اجرا شد!")
            st.write("کل سرمایه‌گذاری مورد نیاز: {:.2f}".format(total_budget_lin))
            
            # نمایش نتایج به صورت جدول
            df_results = pd.DataFrame({
                "بخش": np.arange(1, n + 1),
                "سرمایه‌گذاری": x_opt_lin,
                "رشد مطلق": induced_growth_lin,
                "رشد نسبی": rel_growth_lin
            })
            st.dataframe(df_results)
            
            # رسم نمودار میله‌ای برای رشد نسبی هر بخش با استفاده از arabic_reshaper و python-bidi
            fig1, ax1 = plt.subplots()
            ax1.bar(np.arange(1, n + 1), rel_growth_lin, color='skyblue')
            ax1.set_xlabel(get_display(arabic_reshaper.reshape("بخش")))
            ax1.set_ylabel(get_display(arabic_reshaper.reshape("رشد نسبی")))
            ax1.set_title(get_display(arabic_reshaper.reshape("نمودار رشد نسبی هر بخش")))
            st.pyplot(fig1)
            
            # رسم نمودار دایره‌ای برای تخصیص سرمایه‌گذاری به تفکیک بخش‌ها
            sorted_data = sorted(zip(x_opt_lin, range(1, n + 1)), reverse=True)
            x_sorted, indices_sorted = zip(*sorted_data)
            labels = [get_display(arabic_reshaper.reshape(f"بخش {i}")) for i in indices_sorted]

            fig2, ax2 = plt.subplots()
            ax2.pie(
                x_sorted,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 6}  # تنظیم اندازه فونت برچسب‌ها
            )
            ax2.set_title(
                get_display(arabic_reshaper.reshape("توزیع سرمایه‌گذاری به تفکیک بخش")),
                fontsize=10  # در صورت نیاز، اندازه فونت عنوان را هم می‌توان تغییر داد
            )
            st.pyplot(fig2)
        else:
            st.error("مدل بهینه‌سازی خطی موفق نبود: " + res_lin.message)