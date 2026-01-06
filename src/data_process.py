import random
import string
from PIL import Image, ImageDraw, ImageFont

def wrap_text_by_pixel_width(text, font, max_width):
    """
    按像素宽度精确换行
    
    Args:
        text: 要换行的文本
        font: PIL字体对象
        max_width: 最大像素宽度
    
    Returns:
        lines: 换行后的文本列表
    """
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        # 计算单词的实际像素宽度
        word_width = font.getbbox(word + " ")[2]
        
        # 如果加上这个单词会超出宽度,换行
        if current_width + word_width > max_width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width
        else:
            current_line.append(word)
            current_width += word_width
    
    # 添加最后一行
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines

def render_text_fixed_width(text, font_path, font_size=16, 
                           width=900, padding=20, line_spacing=4):
    """
    固定宽度,高度自适应,精确像素换行
    """
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    
    # 计算可用文本宽度
    text_width = width - 2 * padding
    
    # ✅ 使用像素宽度精确换行
    lines = wrap_text_by_pixel_width(text, font, text_width)
    
    # 计算总高度
    total_height = padding
    for line in lines:
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        total_height += line_height + line_spacing
    total_height = total_height - line_spacing + padding
    
    # 创建图片
    img = Image.new("RGB", (width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 绘制文本
    y_offset = padding
    for line in lines:
        draw.text((padding, y_offset), line, font=font, fill=(0, 0, 0))
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        y_offset += line_height + line_spacing
    
    return img


def make_radom_pic(token_count, tokenizer):
    """生成包含随机文本的图片"""
    # 往图片中写入文本
    # 文本来自使用字母表中的字母随机构造单词
    text = ""
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    # 根据 token_count 生成大约相同数量的单词
    current_token_count = 0
    while current_token_count < token_count:
        # 每个随机单词长度在1到10之间
        word = ''.join(random.choices(alphabet, k=random.randint(1, 10)))
        text += word + " "
        current_token_count = len(tokenizer.encode(text))
    # 将文本写入图片
    img = render_text_fixed_width(
            text=text,
            font_path="../fonts/NotoSans-Regular.ttf",
            font_size=16,
            width=900,
            padding=20,
            line_spacing=4
        )
    return img, current_token_count, text


def distort_text_simple(text, ratio=0.05):
    """
    对文本进行简单扰动，生成两种版本：
    1. 随机交换单词中的两个字母-Swap
    2. 完全打乱单词中的字母顺序-Shuffle
    固定随机种子为0，保证结果可复现。
    Args:
        text: 原始文本
        ratio: 扰动比例，表示要修改的单词占总单词数的比例，5%和10%
    """
    random.seed(0)
    words = text.split()
    n_words = len(words)
    
    # 预处理：分离标点符号，找出所有可以进行操作的单词索引（去除标点后长度大于1）
    valid_indices = []
    parsed_words = [] # 存储 (prefix, core, suffix)
    
    for idx, word in enumerate(words):
        # 分离前缀标点
        prefix = ""
        temp = word
        while temp and temp[0] in string.punctuation:
            prefix += temp[0]
            temp = temp[1:]
        
        # 分离后缀标点
        suffix = ""
        while temp and temp[-1] in string.punctuation:
            suffix = temp[-1] + suffix
            temp = temp[:-1]
            
        core = temp
        parsed_words.append((prefix, core, suffix))
        
        # 只有核心部分长度大于1的才适合进行交换或打乱
        if len(core) > 1:
            valid_indices.append(idx)

    # 计算需要修改的单词数量，至少修改1个（如果文本足够长），或者严格按照比例
    n_modify = int(n_words * ratio)
    if n_modify == 0 and n_words > 0:
        n_modify = 1
        
    if not valid_indices:
        return text, text, {}, {}

    # 从有效索引中随机选择要修改的单词
    indices = random.sample(valid_indices, min(len(valid_indices), n_modify))
    
    # 方式1: 随机交换两个字母
    words_swap = list(words)
    swap_details = {}
    for idx in indices:
        prefix, core, suffix = parsed_words[idx]
        chars = list(core)
        # 随机选两个不同的位置
        i, j = random.sample(range(len(chars)), 2)
        chars[i], chars[j] = chars[j], chars[i]
        new_core = "".join(chars)
        
        new_word = prefix + new_core + suffix
        words_swap[idx] = new_word
        swap_details[idx] = {"original": words[idx], "distorted": new_word}
            
    # 方式2: 完全打乱
    words_shuffle = list(words)
    shuffle_details = {}
    for idx in indices:
        prefix, core, suffix = parsed_words[idx]
        chars = list(core)
        random.shuffle(chars)
        new_core = "".join(chars)
        
        new_word = prefix + new_core + suffix
        words_shuffle[idx] = new_word
        shuffle_details[idx] = {"original": words[idx], "distorted": new_word}
            
    return " ".join(words_swap), " ".join(words_shuffle), swap_details, shuffle_details