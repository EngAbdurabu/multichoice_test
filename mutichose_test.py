import cv2
import numpy as np
import imutils


# دالة لترتيب النقاط (لتحويل المنظور)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # النقطة العليا اليسرى
    rect[2] = pts[np.argmax(s)]  # النقطة السفلى اليمنى
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # النقطة العليا اليمنى
    rect[3] = pts[np.argmax(diff)]  # النقطة السفلى اليسرى
    return rect


# دالة لإجراء تحويل منظور (Perspective Transform)
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # حساب عرض الصورة الجديد
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    # حساب ارتفاع الصورة الجديد
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    # نقاط الوجهة بعد التحويل
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    # حساب مصفوفة التحويل
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# قراءة الصورة الأصلية لورقة الاختبار
image = cv2.imread("omr_test.png")
# تغيير حجم الصورة للراحة في المعالجة
image = imutils.resize(image, height=700)
orig = image.copy()

# تحويل الصورة إلى تدرجات الرمادي
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# تطبيق Gaussian Blur لإزالة الضوضاء
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# تطبيق كشف الحواف باستخدام Canny
edged = cv2.Canny(blurred, 75, 200)

# البحث عن الكونتورز الخارجية
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# العثور على أكبر كونتور رباعي يمثل ورقة الاختبار
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break

if docCnt is None:
    print("لم يتم العثور على ورقة الإجابة")
    exit(0)

# إجراء تحويل منظور للحصول على منظر علوي مستوٍ للورقة
paper = four_point_transform(orig, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# تطبيق thresholding لتحويل الصورة إلى ثنائية (مما يسهل اكتشاف الفقاعات)
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# البحث عن الكونتورز في الصورة المعالجة للعثور على الفقاعات
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

# تحديد الفقاعات باستخدام معايير الحجم والشكل
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    aspectRatio = w / float(h)
    if w >= 20 and h >= 20 and 0.9 <= aspectRatio <= 1.1:
        questionCnts.append(c)

# فرضاً، لنفترض أن الاختبار يحتوي على 5 أسئلة مع 5 خيارات لكل سؤال
questionCnts = sorted(questionCnts, key=lambda c: cv2.boundingRect(c)[1])
questions = []
num_options = 5
for i in range(0, len(questionCnts), num_options):
    cnts_row = sorted(
        questionCnts[i : i + num_options], key=lambda c: cv2.boundingRect(c)[0]
    )
    questions.append(cnts_row)

# مفتاح الإجابات (مثال: رقم السؤال : رقم الخيار الصحيح)
answer_key = {0: 3, 1: 4, 2: 1, 3: 2, 4: 1}
correct = 0

# تحليل كل سؤال وتحديد الإجابة المختارة
for q, cnts_row in enumerate(questions):
    bubbled = None
    max_filled = 0
    for j, c in enumerate(cnts_row):
        # إنشاء قناع (mask) للفقاعة الحالية
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # حساب عدد البيكسلات المملوءة داخل الفقاعة
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        if total > max_filled:
            max_filled = total
            bubbled = j

    # تحديد اللون للتظليل: أخضر للإجابة الصحيحة، أحمر للإجابة الخاطئة
    color = (0, 0, 255)
    k = answer_key.get(q, -1)
    if k == bubbled:
        color = (0, 255, 0)
        correct += 1

    # رسم دائرة حول الفقاعة المختارة
    c = cnts_row[bubbled]
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.circle(paper, (x + w // 2, y + h // 2), 20, color, 2)

# حساب النسبة النهائية للإجابات الصحيحة
score = (correct / len(answer_key)) * 100
print("الدرجة النهائية: {:.2f}%".format(score))
cv2.putText(
    paper,
    "{:.2f}%".format(score),
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 255),
    2,
)

# عرض النتائج النهائية
cv2.imshow("الصورة الأصلية", orig)
cv2.imshow("ورقة الاختبار بعد المعالجة", paper)
cv2.waitKey(0)
cv2.destroyAllWindows()
