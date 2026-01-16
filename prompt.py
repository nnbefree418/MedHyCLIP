# 定义 prompt 模板列表（TEMPLATES）
# 这些模板会作为文本提示（text prompt）用于构造 CLIP 的文本输入，
# 通过格式化 {} 来插入医学图像类别名称，从而生成多样化的句子，
# 实现 prompt ensemble（多提示集成），提升文本编码的稳健性。
TEMPLATES = [
    'a cropped photo of the {}.',                 # 一张裁剪过的该类别的照片
    'a cropped photo of a {}.',                  # 一张裁剪过的某类别对象的照片
    'a close-up photo of a {}.',                 # 一张某类别对象的特写照片
    'a close-up photo of the {}.',               # 一张该类别对象的特写照片
    'a bright photo of a {}.',                   # 一张亮调的某类别照片
    'a bright photo of the {}.',                 # 一张亮调的该类别照片
    'a dark photo of the {}.',                   # 一张暗调的该类别照片
    'a dark photo of a {}.',                     # 一张暗调的某类别照片
    'a jpeg corrupted photo of a {}.',           # 一张 JPEG 损坏过的某类别照片
    'a jpeg corrupted photo of the {}.',         # 一张 JPEG 损坏过的该类别照片
    'a blurry photo of the {}.',                 # 一张模糊的该类别照片
    'a blurry photo of a {}.',                   # 一张模糊的某类别照片
    'a photo of the {}',                         # 一张该类别的照片
    'a photo of a {}',                           # 一张某类别的照片
    'a photo of a small {}',                     # 一张小尺寸某类别对象的照片
    'a photo of the small {}',                   # 一张小尺寸该类别照片
    'a photo of a large {}',                     # 一张大尺寸某类别照片
    'a photo of the large {}',                   # 一张大尺寸该类别照片
    'a photo of the {} for visual inspection.',  # 用于视觉检查的该类别照片
    'a photo of a {} for visual inspection.',    # 用于视觉检查的某类别照片
    'a photo of the {} for anomaly detection.',  # 用于异常检测的该类别照片
    'a photo of a {} for anomaly detection.'     # 用于异常检测的某类别照片
]


# REAL_NAME 字典：将任务名映射为其自然语言描述，
# 会作为 prompt 模板中的 {} 进行填充，从而构造真实语义的文本描述。
# 用于 encode_text_with_prompt_ensemble()，生成 CLIP 文本特征。
REAL_NAME = {
    'Brain': 'Brain',                       # 脑部医学图像
    'Liver':'Liver',                        # 肝脏医学图像
    'Retina_RESC':'retinal OCT',            # 视网膜 OCT 图像（RESC 数据集）
    'Chest':'Chest X-ray film',             # 胸部 X 光片
    'Retina_OCT2017':'retinal OCT',         # 视网膜 OCT 图像（OCT2017 数据集）
    'Histopathology':'histopathological image'  # 病理组织学图像
}
