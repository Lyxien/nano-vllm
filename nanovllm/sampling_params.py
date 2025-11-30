from dataclasses import dataclass


# 对于生成的多个结果 token 的采样配置
@dataclass
class SamplingParams:
    # 采样的多样性
    temperature: float = 1.0
    # 最多生成 token 的数量（仅限制输出的token数量）
    max_tokens: int = 64
    # 是否忽略模型生成的 </eos>（结束符）
    # False（默认），一旦模型预测到 eos → 停止生成
    # 即使预测到 eos，也继续生成到 max_tokens 为止
    # 适用场景：想生成多段文本、训练数据 eos 嵌在中间、做续写，但 prompt 里含 eos
    ignore_eos: bool = False
