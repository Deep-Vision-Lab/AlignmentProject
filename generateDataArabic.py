# arabic_image_generator.py
# Description: Generates pairs of images with Arabic text and their corresponding
#              Smith-Waterman alignment scoring matrices.
# Dependencies: Pillow, arabic-reshaper, python-bidi, tqdm, numpy, re
# To install: pip install Pillow arabic-reshaper python-bidi tqdm numpy re

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
import random
from Parameters import *
import os
import re

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from DiffNWAlgo import DiffNWAlgo

import warnings
warnings.filterwarnings("ignore")


def needleman_wunsch_matrix(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    score_matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(cols):
        score_matrix[0, j] = j * gap_penalty
    for i in range(1, rows):
        for j in range(1, cols):
            similarity = match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty
            diagonal_score = score_matrix[i - 1, j - 1] + similarity
            up_score = score_matrix[i - 1, j] + gap_penalty
            left_score = score_matrix[i, j - 1] + gap_penalty
            score_matrix[i, j] = max(diagonal_score, up_score, left_score)
    return score_matrix[1:, 1:]


def remove_diacritics(text):
    arabic_diacritics = re.compile("""
                                 ّ    | # Shadda
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
    return re.sub(arabic_diacritics, '', text)


def save_matrix_to_file(matrix, filepath):
    try:
        np.save(filepath, matrix)
    except Exception as e:
        print(f"Error saving matrix to {filepath}: {e}")


def create_similarity_matrix(seq1, seq2):
    rows = len(seq1)
    cols = len(seq2)
    similarity_matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if seq1[i] == seq2[j]:
                similarity_matrix[i, j] = 1
    return similarity_matrix


def create_arabic_text_image(text_to_render, font_path_or_name, font_size_px, image_dimensions_px, output_image_path,
                             output_mask_path=None, substring_to_mask=None,
                             text_color="white", background_color="black", padding=20):
    reshaped_text = arabic_reshaper.reshape(text_to_render)
    bidi_text = get_display(reshaped_text)
    bidi_text = remove_diacritics(bidi_text)

    try:
        font = ImageFont.truetype(font_path_or_name, font_size_px)
    except IOError:
        print(f"Critical: Fallback fonts not found for {output_image_path}. Using PIL's default font.")
        font = ImageFont.load_default()

    image = Image.new("RGB", (1000, 1000), background_color)
    draw = ImageDraw.Draw(image)

    left, top, right, bottom = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = right - left
    text_height = bottom - top

    image_width = int(text_width + padding * 2)
    image_height = int(text_height + padding * 2)

    image = Image.new('RGB', (image_width, image_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    position = (padding, padding)
    draw.text(position, bidi_text, (255, 255, 255), font=font)
    image = image.resize(image_dimensions_px)
    image.save(output_image_path)

    if output_mask_path and substring_to_mask:
        mask_image = Image.new('L', (image_width, image_height), color=0)
        mask_draw = ImageDraw.Draw(mask_image)

        # Search in the original text (before reshaping) so character forms don't matter.
        # Reshaping changes joining forms based on context, making find() on bidi text fail.
        phrase_clean = remove_diacritics(substring_to_mask)
        text_clean = remove_diacritics(text_to_render)
        start_idx = text_clean.find(phrase_clean)
        if start_idx != -1:
            end_idx = start_idx + len(phrase_clean)
            n = len(bidi_text)
            # For RTL text, logical positions [start:end] map to visual positions [n-end:n-start]
            vis_start = max(0, n - end_idx)
            vis_end = min(n, n - start_idx)
            prefix = bidi_text[:vis_start]
            matched = bidi_text[vis_start:vis_end]
            x_offset = padding + draw.textlength(prefix, font=font)
            sub_width = draw.textlength(matched, font=font)
            mask_draw.rectangle([x_offset, 0, x_offset + sub_width, image_height], fill=255)
        else:
            print(f"Warning: mask substring not found in `{output_image_path}`")

        mask_image = mask_image.resize(image_dimensions_px)
        mask_image.save(output_mask_path)


def save_text_to_file(text, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving text to {filepath}: {e}")


def get_augment_phrase(phrases_list, empty_prob):
    if not phrases_list:
        return ""
    if random.random() < empty_prob:
        return ""
    return random.choice(phrases_list)


if __name__ == "__main__":
    num_samples_to_generate = 100000
    base_output_directory = f"DataSet/Synthetic_{lang}_{num_samples_to_generate}"
    EMPTY_AUGMENT_PROBABILITY = 0.2

    output_images_dir = os.path.join(base_output_directory, "images")
    output_masks_dir = os.path.join(base_output_directory, "masks")
    output_matrices_dir = os.path.join(base_output_directory, "matrices")
    output_similarity_matrices_dir = os.path.join(base_output_directory, "similarity_matrices")
    output_diff_nw_matrices_dir = os.path.join(base_output_directory, "diff_nw_matrices")
    output_text_lines_dir = os.path.join(base_output_directory, "texts")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_matrices_dir, exist_ok=True)
    os.makedirs(output_similarity_matrices_dir, exist_ok=True)
    os.makedirs(output_diff_nw_matrices_dir, exist_ok=True)
    os.makedirs(output_text_lines_dir, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(base_output_directory)}")

    base_arabic_phrases = [
        "الشمس مشرقة اليوم حقاً",
        "القهوة صباحاً تعدل المزاج",
        "الحديقة مليئة بالزهور الملونة",
        "القراءة غذاء العقل والروح",
        "السفر يفتح آفاقاً جديدة",
        "الرياضة تحافظ على صحة الجسم",
        "الموسيقى الهادئة تريح الأعصاب",
        "العمل الجاد يؤتي ثماره دائماً",
        "التكنولوجيا تسهل حياتنا كثيراً",
        "التعاون يحقق أفضل النتائج الممكنة",
        "الأزهار الجميلة تتفتح في الربيع",
        "البحر يبدو اليوم جميلاً وهادئاً",
        "التاريخ يعلمنا دروساً قيمة ومهمة",
        "البيئة الطبيعية تحتاج إلى رعايتنا جميعاً",
        "الأمل الصادق يعطينا القوة للاستمرار",
        "القصص المسلية ممتعة جداً للأطفال",
        "أوراق الشجر الخضراء تتساقط خريفاً",
        "العائلة هي السند الحقيقي وقت الشدة",
        "الابتسامة الصادقة تنشر السعادة حولنا",
        "النجوم اللامعة تلمع في السماء ليلاً",
        "الأحلام الكبيرة تحتاج إلى سعي وجهد",
        "الفصول الأربعة تتغير بانتظام دائماً",
        "الطعام العربي التقليدي لذيذ جداً",
        "الاستماع الجيد يعتبر مهارة مهمة",
        "الصداقة الحقيقية كنز ثمين بالفعل",
        "الجو لطيف ومشمس هذا الصباح",
        "الكتاب الجديد يبدو مثيراً للاهتمام",
        "الهدوء يعم المكان الآن",
        "الأخبار السارة أسعدت الجميع",
        "الوقت يمر بسرعة كبيرة جداً",
        "المطر يهطل بغزارة في الخارج",
        "الأطفال يلعبون في الحديقة بمرح",
        "الشاي الدافئ يريح النفس في البرد",
        "العلم نور يضيء طريق المستقبل",
        "الكرم والجود من أجمل الصفات الإنسانية",
        "المدينة تبدو جميلة عند الغروب",
        "الطيور تغني في الصباح الباكر",
        "الأمانة أساس كل علاقة ناجحة",
        "التواضع صفة تزيد المرء رفعة",
        "الحب والتفاهم يبنيان الأسر السعيدة",
        "الطبيعة تمنحنا الهواء النقي والماء",
        "الرحلات البعيدة تعلمنا ثقافات جديدة",
        "التفكير الإيجابي يغير حياة الإنسان",
        "الصبر مفتاح الفرج في كل الأمور",
        "التعليم يرفع مستوى المجتمعات النامية",
        "الحضارة الإنسانية تبنى بالعلم والعمل",
        "الأقارب الكرام يجلبون الفرح والسرور",
        "الجبال الشامخة تروي قصص الطبيعة",
        "النهر يجري بهدوء نحو البحر البعيد",
        "الخضروات الطازجة مفيدة للصحة كثيراً",
        "الأشجار الكبيرة توفر الظل في الصيف",
        "الأفكار الجديدة تفتح آفاق الإبداع",
        "الشعر العربي يعبر عن أعمق المشاعر",
        "المساجد تمتلئ بالمصلين في رمضان",
        "الورد الأحمر يفوح برائحة زكية جميلة",
        "التضحية من أجل الوطن واجب مقدس",
        "الأمهات ينثرن الحنان على أبنائهن",
        "الهواء المنعش يجدد النشاط والطاقة",
        "الحدائق العامة تجمع الناس معاً",
        "القمر يضيء الليل بنور هادئ جميل",
        "الألوان الزاهية تبهج النفس وتسعدها",
        "الرياح الباردة تعصف في الشتاء القارس",
        "المكتبة العامة مصدر المعرفة والثقافة",
        "الزيارات العائلية تجدد روابط المحبة",
        "الإبداع يزدهر في بيئة حرة مشجعة",
        "الكلمة الطيبة صدقة تبقى في القلوب",
        "الولد الصغير يتعلم من كل تجربة",
        "السكينة تسكن قلب المؤمن الصادق",
        "البسمة الدافئة تذيب جليد القلوب",
        "الحرية قيمة لا تقدر بثمن أبداً",
        "المدرسة بيت العلم والتربية والأخلاق",
    ]

    augmenting_arabic_phrases = [
        "بكل تأكيد",
        "في بعض الأحيان",
        "بشكل عام",
        "على الأغلب الظن",
        "في الواقع تماماً",
        "لهذا السبب بالذات",
        "مع الأسف الشديد",
        "لحسن الحظ فعلاً",
        "بعد قليل من الوقت",
        "منذ فترة ليست بالطويلة",
        "حتى الآن فقط",
        "في المساء الباكر",
        "قبل الظهر مباشرة",
        "أثناء النهار كله",
        "ببطء شديد وحذر",
        "بسرعة كبيرة جداً",
        "بدون أدنى شك",
        "على الإطلاق أبداً",
        "مرة أخرى قريباً",
        "في هذا المكان الجميل",
        "هناك أيضاً سبب آخر",
        "على سبيل المثال لا الحصر",
        "فوق كل شيء آخر",
        "تحت أي ظرف كان",
        "بجوار النافذة الكبيرة",
        "في الحقيقة",
        "غالباً ما",
        "ربما لاحقاً",
        "بالتأكيد نعم",
        "خصوصاً اليوم",
        "بعيداً عن الضوضاء",
        "وسط الطبيعة الهادئة",
        "مع طلوع الفجر",
        "في وقت قصير",
        "بجهد متواصل",
        "بفضل الله أولاً",
        "بعد تفكير عميق",
        "منذ زمن بعيد",
        "قريباً جداً الآن",
        "تحت سماء صافية",
        "بين أهل الخير",
        "في ظل الظروف",
        "بصوت منخفض هادئ",
        "بكل هدوء وأناة",
        "في لحظة مناسبة",
        "بنظرة متفائلة دائماً",
        "بعد مسيرة طويلة",
        "على مدار اليوم",
        "دون انقطاع أبداً",
        "بكثير من الصبر",
        "في الوقت المحدد",
        "بطريقة ذكية ومنظمة",
        "تدريجياً وبهدوء",
        "مع مرور الأيام",
        "في أعماق القلب",
        "خلف الأفق البعيد",
        "بأسلوب جميل ومميز",
        "وفق الخطة الموضوعة",
        "حتى آخر لحظة",
        "بقلب راضٍ وشاكر",
        "في الصمت المطبق",
        "بعين ثاقبة وواعية",
        "رغم الصعاب الكثيرة",
        "وسط الزحام الشديد",
        "بعد طول انتظار",
        "بكامل الاهتمام والعناية",
        "في أوقات الفراغ",
        "مع نسيم الصباح",
        "قبل حلول الظلام",
        "بلا توقف أو كلل",
    ]

    font_to_use = "Fonts/Amiri-Regular.ttf"
    text_font_size = 90
    img_width = 1024
    img_height = 128
    image_dimensions = (img_width, img_height)

    NW_match_score = matchScore
    NW_mismatch_penalty = mismatchScore
    NW_gap_penalty = gapScore

    diff_nw_algo = DiffNWAlgo(match_score=matchScore, miss_score=mismatchScore, gap=gapScore, batch=False)

    TARGET_LEN = 63

    # ── Phase 1: generate all sentence pairs and collect them ──────────────────
    print(f"\nPhase 1: generating {num_samples_to_generate} sentence pairs (exactly {TARGET_LEN} letters each)...")
    samples = []  # list of (sentence1, sentence2, common_phrase)

    with tqdm(total=num_samples_to_generate, desc="Building sentences") as pbar:
     while len(samples) < num_samples_to_generate:
        alignment_strategy = random.choice([
            'start_end', 'start_start', 'end_end', 'middle_middle', 'end_start',
            'middle_start', 'middle_end', 'start_middle', 'end_middle'
        ])

        common_phrase = random.choice(base_arabic_phrases)
        num_aug_parts = 2

        aug_parts1 = [get_augment_phrase(augmenting_arabic_phrases, EMPTY_AUGMENT_PROBABILITY) for _ in range(num_aug_parts)]
        aug_parts2 = [get_augment_phrase(augmenting_arabic_phrases, EMPTY_AUGMENT_PROBABILITY) for _ in range(num_aug_parts)]
        aug_parts3 = [get_augment_phrase(augmenting_arabic_phrases, EMPTY_AUGMENT_PROBABILITY) for _ in range(num_aug_parts)]
        aug_parts4 = [get_augment_phrase(augmenting_arabic_phrases, EMPTY_AUGMENT_PROBABILITY) for _ in range(num_aug_parts)]

        while not any(p.strip() for p in aug_parts1):
            aug_parts1 = [random.choice(augmenting_arabic_phrases) for _ in range(num_aug_parts)]
        while not any(p.strip() for p in aug_parts2):
            aug_parts2 = [random.choice(augmenting_arabic_phrases) for _ in range(num_aug_parts)]
        while not any(p.strip() for p in aug_parts3):
            aug_parts3 = [random.choice(augmenting_arabic_phrases) for _ in range(num_aug_parts)]
        while not any(p.strip() for p in aug_parts4):
            aug_parts4 = [random.choice(augmenting_arabic_phrases) for _ in range(num_aug_parts)]

        if alignment_strategy == 'start_end':
            sentence1_parts = [common_phrase] + aug_parts1
            sentence2_parts = aug_parts2 + [common_phrase]
        elif alignment_strategy == 'start_start':
            sentence1_parts = [common_phrase] + aug_parts1
            sentence2_parts = [common_phrase] + aug_parts2
        elif alignment_strategy == 'end_end':
            sentence1_parts = aug_parts1 + [common_phrase]
            sentence2_parts = aug_parts2 + [common_phrase]
        elif alignment_strategy == 'end_start':
            sentence1_parts = aug_parts1 + [common_phrase]
            sentence2_parts = [common_phrase] + aug_parts2
        elif alignment_strategy == 'middle_start':
            sentence1_parts = aug_parts1 + [common_phrase] + aug_parts2
            sentence2_parts = [common_phrase] + aug_parts3
        elif alignment_strategy == 'middle_end':
            sentence1_parts = aug_parts1 + [common_phrase] + aug_parts2
            sentence2_parts = aug_parts3 + [common_phrase]
        elif alignment_strategy == 'start_middle':
            sentence1_parts = [common_phrase] + aug_parts1
            sentence2_parts = aug_parts2 + [common_phrase] + aug_parts3
        elif alignment_strategy == 'end_middle':
            sentence1_parts = aug_parts1 + [common_phrase]
            sentence2_parts = aug_parts2 + [common_phrase] + aug_parts3
        else:  # middle_middle
            sentence1_parts = aug_parts1 + [common_phrase] + aug_parts2
            sentence2_parts = aug_parts3 + [common_phrase] + aug_parts4

        s1 = remove_diacritics(" ".join(part.strip() for part in sentence1_parts if part.strip()))
        s2 = remove_diacritics(" ".join(part.strip() for part in sentence2_parts if part.strip()))

        if len(s1) < TARGET_LEN or len(s2) < TARGET_LEN:
            continue
        s1 = s1[:TARGET_LEN]
        s2 = s2[:TARGET_LEN]
        if s1[-1] == ' ' or s2[-1] == ' ':
            continue
        common_phrase_clean = remove_diacritics(common_phrase)
        if common_phrase_clean not in s1 or common_phrase_clean not in s2:
            continue

        samples.append((s1, s2, common_phrase))
        pbar.update(1)

    # ── Phase 2: save images, matrices, and text files ────────────────────────
    print(f"\nPhase 2: saving outputs...")

    for i, (s1, s2, common_phrase) in enumerate(tqdm(samples, desc="Saving outputs"), 1):
        output_img_file_1 = os.path.join(output_images_dir, f"img1_{i}.png")
        output_img_file_2 = os.path.join(output_images_dir, f"img2_{i}.png")
        output_mask_file_1 = os.path.join(output_masks_dir, f"mask1_{i}.png")
        output_mask_file_2 = os.path.join(output_masks_dir, f"mask2_{i}.png")
        output_matrix_file = os.path.join(output_matrices_dir, f"scoreMatrix_{i}.npy")
        output_similarity_matrix_file = os.path.join(output_similarity_matrices_dir, f"similarityMatrix_{i}.npy")
        output_diff_nw_matrix_file = os.path.join(output_diff_nw_matrices_dir, f"diffNWMatrix_{i}.npy")
        output_text_file_1 = os.path.join(output_text_lines_dir, f"text1_{i}.txt")
        output_text_file_2 = os.path.join(output_text_lines_dir, f"text2_{i}.txt")

        # Images: use original (unpadded) text; resize handles fixed dimensions
        create_arabic_text_image(s1, font_to_use, text_font_size, image_dimensions,
                                 output_img_file_1, output_mask_file_1, common_phrase)
        create_arabic_text_image(s2, font_to_use, text_font_size, image_dimensions,
                                 output_img_file_2, output_mask_file_2, common_phrase)

        # Matrices: use space-stripped original (padding must not affect alignment)
        seq1_no_spaces = s1.replace(" ", "")
        seq2_no_spaces = s2.replace(" ", "")

        scoring_matrix = needleman_wunsch_matrix(
            seq1_no_spaces, seq2_no_spaces,
            match_score=NW_match_score,
            mismatch_penalty=NW_mismatch_penalty,
            gap_penalty=NW_gap_penalty
        )
        save_matrix_to_file(scoring_matrix, output_matrix_file)

        similarity_matrix = create_similarity_matrix(seq1_no_spaces, seq2_no_spaces)
        save_matrix_to_file(similarity_matrix, output_similarity_matrix_file)

        sim_tensor = torch.tensor(similarity_matrix, dtype=torch.float32)
        scaled_sim = sim_tensor * (matchScore - mismatchScore) + mismatchScore

        with torch.no_grad():
            diff_nw_align = diff_nw_algo(similarity_matrix=scaled_sim, calc_cosine=False)

        diff_nw_matrix = diff_nw_align.squeeze(0).cpu().numpy()
        save_matrix_to_file(diff_nw_matrix, output_diff_nw_matrix_file)

        save_text_to_file(s1, output_text_file_1)
        save_text_to_file(s2, output_text_file_2)

    print(
        f"\nDone. {num_samples_to_generate} samples generated in '{base_output_directory}'.\n"
        f"All text files: {TARGET_LEN} characters each."
    )
