import cv2
import numpy as np
import imutils
import tkinter as tk
from tkinter import filedialog
import os


def order_points(pts):
    """
    هذه الدالة ترتب النقاط الأربع بحيث نعرف أي نقطة تقع في الأعلى واليسار، وأيها في الأسفل واليمين.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)  # نجمع إحداثيات كل نقطة
    rect[0] = pts[
        np.argmin(s)
    ]  # النقطة التي يكون مجموعها الأصغر ستكون في الأعلى واليسار
    rect[2] = pts[
        np.argmax(s)
    ]  # النقطة التي يكون مجموعها الأكبر ستكون في الأسفل واليمين
    diff = np.diff(pts, axis=1)  # نحسب الفرق بين الإحداثيات
    rect[1] = pts[
        np.argmin(diff)
    ]  # النقطة التي يكون فرقها الأصغر تكون في الأعلى واليمين
    rect[3] = pts[
        np.argmax(diff)
    ]  # النقطة التي يكون فرقها الأكبر تكون في الأسفل واليسار
    return rect


def four_point_transform(image, pts):
    """
    هذه الدالة تقوم بتعديل الصورة بحيث تظهر وكأننا ننظر إليها من فوق بشكل مستقيم.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # نحسب العرض الجديد للصورة باستخدام المسافة بين النقاط
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # نحسب الارتفاع الجديد للصورة
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # نحدد النقاط التي ستذهب إليها زوايا الصورة بعد التحويل
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # نحسب مصفوفة التحويل ونطبقها على الصورة
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def process_exam_sheet(
    image_path, answer_key, num_questions=5, num_options=5, output_folder="output"
):
    """
    هذه الدالة تقوم بمعالجة صورة ورقة الاختبار:
    - قراءة الصورة وتعديل حجمها.
    - تحويلها إلى صورة رمادية وتنعيمها.
    - اكتشاف الحواف والبحث عن حدود ورقة الاختبار.
    - تصحيح المنظور باستخدام تحويل (Perspective Transform).
    - تطبيق adaptive thresholding لتحويل الصورة إلى أبيض وأسود.
    - اكتشاف الفقاعات وتحليل كل سؤال لتحديد الإجابة المختارة.
    - مقارنة الإجابة المختارة مع الإجابة الصحيحة التي أدخلها المستخدم.
    - عرض النتيجة النهائية وحفظ الصورة.
    """
    # 1. قراءة الصورة
    image = cv2.imread(image_path)
    if image is None:
        print(f"Sorry We cann't read Image: {image_path}")
        return
    image = imutils.resize(image, height=700)  # تغيير الحجم لتسهيل المعالجة
    orig = image.copy()  # حفظ نسخة من الصورة الأصلية

    # 2. تحويل الصورة إلى تدرجات الرمادي وتنعيمها
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)  # اكتشاف الحواف

    # 3. البحث عن الكونتورز الخارجية لإيجاد حدود ورقة الاختبار
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    if docCnt is None:
        print("We cann't find the image:", image_path)
        return

    # 4. إجراء تحويل منظور للحصول على ورقة الاختبار بشكل مسطح
    paper = four_point_transform(orig, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # 5. تطبيق Adaptive Thresholding لتحويل الصورة إلى أبيض وأسود حتى مع اختلاف الإضاءة
    thresh = cv2.adaptiveThreshold(
        warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 6. اكتشاف الكونتورز التي تمثل الفقاعات
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        if w >= 20 and h >= 20 and 0.9 <= aspectRatio <= 1.1:
            questionCnts.append(c)

    # ترتيب الكونتورز من الأعلى إلى الأسفل
    questionCnts = sorted(questionCnts, key=lambda c: cv2.boundingRect(c)[1])
    questions = []
    # تقسيم الكونتورز إلى أسئلة بناءً على عدد الخيارات (مثلاً 5 خيارات لكل سؤال)
    for i in range(0, len(questionCnts), num_options):
        cnts_row = sorted(
            questionCnts[i : i + num_options], key=lambda c: cv2.boundingRect(c)[0]
        )
        questions.append(cnts_row)

    correct = 0  # عداد الإجابات الصحيحة

    # 7. تحليل كل سؤال:
    for q, cnts_row in enumerate(questions):
        bubbled = None  # متغير لتحديد أي فقاعة تم اختيارها
        max_filled = 0  # لتخزين عدد البكسلات البيضاء في الفقاعة المختارة
        # نفحص كل خيار في السؤال
        for j, c in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            if total > max_filled:
                max_filled = total
                bubbled = j

        # نختار لون الدائرة: أخضر إذا كانت الإجابة صحيحة، أحمر إذا كانت خاطئة
        color = (0, 0, 255)  # أحمر
        correct_answer = answer_key.get(q, -1)
        if bubbled == correct_answer:
            color = (0, 255, 0)  # أخضر
            correct += 1

        # رسم دائرة حول الفقاعة المختارة
        c = cnts_row[bubbled]
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.circle(paper, (x + w // 2, y + h // 2), 20, color, 2)

    # 8. حساب وعرض النتيجة النهائية على الصورة
    score = (correct / float(len(answer_key))) * 100
    cv2.putText(
        paper,
        f"Score: {score:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
    )

    # عرض الصورة النهائية على الشاشة
    cv2.imshow("Processed Exam Sheet", paper)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # حفظ الصورة النهائية في مجلد الإخراج
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, paper)
    print(f"The final image save as: {output_path}")


def main():
    """
    الدالة الرئيسية التي تنسق عملية اختيار الصور وإدخال الإجابات ومعالجة كل صورة.
    """
    # استخدام Tkinter لفتح نافذة اختيار الملفات للسماح للمستخدم باختيار أكثر من صورة
    root = tk.Tk()
    root.withdraw()  # إخفاء النافذة الرئيسية
    file_paths = filedialog.askopenfilenames(
        title="Chose the Image or exam paper: ",
        filetypes=(("Image Files", "*.jpg *.jpeg *.png"),),
    )

    if not file_paths:
        print("There are no image choosed")
        return

    # طلب إدخال الإجابات الصحيحة لكل سؤال
    # "الرجاء إدخال رقم الإجابة الصحيحة لكل سؤال مفصولة بفاصلة (مثال: 1,4,0,3,1)"
    answers_input = input(
        "Enter the correct answer sparate by comma (Ex: 1, 2,5,6,0): "
    )
    answers_list = [int(ans.strip()) for ans in answers_input.split(",")]
    # بناء قاموس الإجابات بحيث يكون مفتاح السؤال ورقم الإجابة الصحيحة
    answer_key = {i: ans for i, ans in enumerate(answers_list)}

    # معالجة كل صورة تم اختيارها
    for image_path in file_paths:
        print(f"image processing: {image_path}")
        process_exam_sheet(image_path, answer_key)

    print("The image processing is done.")


if __name__ == "__main__":
    main()
