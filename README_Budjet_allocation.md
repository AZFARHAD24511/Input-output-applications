<div dir="rtl">

### بهینه‌سازی تخصیص بودجه استان‌ها بر اساس جدول داده-ستانده منطقه‌ای

در پاسخ به نیاز روزافزون سازمان‌های مدیریت و برنامه‌ریزی استان‌ها برای تخصیص بهینه بودجه بین بخش‌های مختلف اقتصادی — با این قید که حداقل نرخ رشد اقتصادی مشخصی (مثلاً ۶ درصد) محقق شود و هم‌زمان طیف وسیعی از بخش‌های اقتصادی را در بر گیرد — برنامه‌ای طراحی کرده‌ام که دقیقاً به همین منظور توسعه یافته است.

این برنامه بر پایه‌ی جدول داده-ستانده منطقه‌ای عمل می‌کند و دو خروجی اصلی ارائه می‌دهد:

1. تعیین حداقل بودجه لازم برای دستیابی به رشد اقتصادی هدف‌گذاری‌شده؛  
2. شناسایی اینکه کدام بخش‌های اقتصادی در این تخصیص، چه میزان رشد اسمی در ارزش افزوده خواهند داشت.

در این برنامه، نرخ رشد اسمی ارزش افزوده و نرخ تورم مورد نظر توسط کاربر وارد می‌شود و سپس برنامه با استفاده از منطق حسابداری جدول داده-ستانده، پارامترهای لازم را استخراج و محاسبه می‌نماید.  
این ابزار به برنامه‌ریزان کمک می‌کند تا تصمیم‌گیری‌های بودجه‌ای را به‌صورت علمی و مبتنی بر داده انجام دهند.

---

## 📐 مدل ریاضی بهینه‌سازی رشد اقتصادی مبتنی بر جدول داده-ستانده

### 🔹 تعریف نمادها:

| نماد | توضیح |
|------|--------|
| $L$ | ماتریس معکوس لئونتیف (Leontief Inverse) |
| $A$ | ماتریس ضرایب فنی بین‌بخشی |
| $\vec{g}$ | بردار ارزش افزوده هر بخش (GVA) |
| $\vec{x}$ | بردار سرمایه‌گذاری بهینه در هر بخش |
| $\epsilon$ | نرخ رشد حداقلی برای هر بخش (ناشی از تورم و رشد واقعی) |
| $G_{\text{target}}$ | رشد اسمی کل مورد انتظار در اقتصاد |
| $\vec{w}_{\text{cost}}$ | بردار ضرایب وزن هزینه‌ای برای سرمایه‌گذاری در هر بخش |

---

### 🔸 تابع هدف:

**کمینه‌سازی مجموع هزینه‌های سرمایه‌گذاری وزن‌دار:**

$$
\min_{\vec{x} \geq 0} \quad \vec{w}_{\text{cost}}^\top \vec{x}
$$

---

### 🔸 قیدها:

1. **رشد نسبی کل اقتصاد باید حداقل به سطح هدف برسد:**

$$
\sum_i (L \vec{x})_i \geq G_{\text{target}} = \frac{\text{TargetGrowthPercent}}{100} \times \sum_i g_i
$$



2. **رشد هر بخش باید حداقل برابر با $\epsilon \cdot g_i$ باشد:**

$$
(L \vec{x})_i \geq \epsilon \cdot g_i \quad \forall i = 1, 2, \dots, n
$$

---

### 🔸 وزن‌دهی تابع هدف:

$$
w_i = 1 + \lambda_1 (1 - BL_i^{\text{norm}}) + \lambda_2 (1 - FL_i^{\text{norm}}) + \mu_1 \cdot CV_{BL_i}^{\text{norm}} + \mu_2 \cdot CV_{FL_i}^{\text{norm}}
$$

که در آن:

* $BL_i = \sum_j L_{ij}$ → شاخص لینکج پسین  
* $FL_i = \sum_j G_{ij}$ → شاخص لینکج پیشین با استفاده از معکوس گُش: $G = (I - A^\top)^{-1}$  
* $A$  ضریب تغییرات سطر/ستون از $CV$  
* پارامترهای $\lambda_1, \lambda_2, \mu_1, \mu_2$ ضرایب قابل تنظیم هستند.

---

### 🔸 خروجی‌ها:

پس از حل مدل:

* $\vec{x}^{\ast}$ → سرمایه‌گذاری بهینه در هر بخش  
* $L \vec{x}^{\ast}$ → رشد مطلق تولید هر بخش  
* $\frac{L \vec{x}^{\ast}}{\vec{g}}$ → رشد نسبی هر بخش  

---

### 🔸 نمونه کار برای استان هرمزگان در پلتفرم Streamlit:

[![allocation](https://github.com/AZFARHAD24511/IO_Budget/blob/main/budget_allocation1.png)](https://iobudget-pdv2ak9zuxbmqyq485jf7v.streamlit.app/)

---

### 🔸 نمونه کار برای استان هرمزگان در پلتفرم Shiny (این وب‌سایت در داخل ایران قابل دسترسی است):

[![allocation](https://github.com/AZFARHAD24511/IO_Budget/blob/main/shiny_io_hormz.png)](https://azfar2451.shinyapps.io/hormozgan-shiny/)

---

### 🔸 فایل داده نمونه:

برای اجرای این برنامه، به یک جدول داده‌-ستانده در قالب فایل Excel نیاز دارید.  
نمونه‌ای از این فایل را می‌توانید از لینک زیر دریافت کنید:

[Industry_by_Industry_IOT_Model_D.xlsx](https://github.com/AZFARHAD24511/datasets/raw/refs/heads/main/Industry_by_Industry_IOT_Model_D.xlsx)

</div>

---

### Budget Allocation Optimization Based on Regional Input-Output Tables

In response to the growing demand from provincial Planning and Budget Organizations regarding the optimal allocation of budgets among economic sectors — under the constraint of achieving a minimum economic growth rate (e.g., 6%) while ensuring broad sectoral participation — I have developed a program tailored to this need.

This application is built upon the regional input-output (I-O) table and provides two core insights:

1. It calculates the minimum budget required to reach the specified economic growth.  
2. It identifies which economic sectors will experience nominal value-added growth under this allocation.

Users input the desired nominal growth rate of value added and the target inflation rate, and the program, using input-output accounting principles, computes the necessary parameters.  
This allows for evidence-based planning and supports effective budgeting decisions across sectors.

---

**To run the app, you need an input-output table in Excel format, such as the one available at the following link:**  
[Industry_by_Industry_IOT_Model_D.xlsx](https://github.com/AZFARHAD24511/datasets/raw/refs/heads/main/Industry_by_Industry_IOT_Model_D.xlsx)
