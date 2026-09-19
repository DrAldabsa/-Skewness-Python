import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, gaussian_kde

# تثبيت النتائج
np.random.seed(42)

n = 1000

# 1. التوزيع الطبيعي
normal_data = np.random.normal(
    loc=50,
    scale=10,
    size=n
)

# 2. التوزيع المنحرف جهة اليمين
right_skewed_data = np.random.gamma(
    shape=2.0,
    scale=8.0,
    size=n
) + 30

# 3. التوزيع المنحرف جهة اليسار
left_skewed_data = 110 - right_skewed_data

datasets = [
    ("Normal Distribution", normal_data),
    ("Right-Skewed Distribution", right_skewed_data),
    ("Left-Skewed Distribution", left_skewed_data)
]

# إنشاء الشكل
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5)
)

fig.suptitle(
    "Effect of Skewness on Data Distribution",
    fontsize=18,
    fontweight="bold"
)

# رسم التوزيعات
for ax, (title, data) in zip(axes, datasets):

    # حساب الإحصاءات
    mean_value = np.mean(data)
    median_value = np.median(data)
    skewness_value = skew(data)

    # Histogram
    ax.hist(
        data,
        bins=30,
        density=True,
        alpha=0.65,
        edgecolor="black"
    )

    # منحنى الكثافة KDE
    kde = gaussian_kde(data)

    x = np.linspace(
        data.min(),
        data.max(),
        400
    )

    ax.plot(
        x,
        kde(x),
        linewidth=2
    )

    # خط المتوسط
    ax.axvline(
        mean_value,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.2f}"
    )

    # خط الوسيط
    ax.axvline(
        median_value,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_value:.2f}"
    )

    # العنوان
    ax.set_title(
        f"{title}\n"
        f"Skewness = {skewness_value:.2f}",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Values")
    ax.set_ylabel("Density")

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend(
        fontsize=9
    )

# تنسيق الشكل
plt.tight_layout(
    rect=[0, 0, 1, 0.92]
)

# حفظ الصورة بدقة عالية
plt.savefig(
    "skewness_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# طباعة النتائج
for title, data in datasets:

    print("\n" + title)
    print("-" * 40)

    print(f"Mean: {np.mean(data):.3f}")

    print(f"Median: {np.median(data):.3f}")

    print(
        f"Standard Deviation: "
        f"{np.std(data, ddof=1):.3f}"
    )

    print(
        f"Skewness: "
        f"{skew(data):.3f}"
    )