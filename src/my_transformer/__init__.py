"""自实现 Pre-LN Transformer 的公开接口；导入时不执行训练。

包外可直接导入所需类和 mask；包内模块使用具体文件的相对导入，避免循环依赖。
这里仅重导出符号，不创建模型、不读取 checkpoint，也不修改模型参数。
"""

# 1. 完整模型与两侧堆叠：负责把词编号转换为目标词表分数。
from .transformer import Transformer
from .encoder import Encoder
from .decoder import Decoder
# 2. 单层 Block 与位置编码：组织 Pre-LN、残差和子层之间的顺序。
from .encoder_layer import EncoderLayer
from .decoder_layer import DecoderLayer
from .positional_encoding import PositionalEncoding
# 3. 底层计算分支：不自行添加 Encoder/Decoder 的残差与 LayerNorm。
from .multi_head_attention import MultiHeadAttention
from .positionwise_feed_forward import PositionwiseFeedForward
from .scaled_dot_product_attention import ScaledDotProductAttention
# 4. 屏蔽辅助函数：本包约定 True 表示允许关注。
from .masks import get_pad_mask, get_subsequent_mask

# 明确公开 API，不改变各类的构造参数或 forward 返回格式。
__all__ = [
    'Transformer',
    'Encoder',
    'Decoder',
    'EncoderLayer',
    'DecoderLayer',
    'PositionalEncoding',
    'MultiHeadAttention',
    'PositionwiseFeedForward',
    'ScaledDotProductAttention',
    'get_pad_mask',
    'get_subsequent_mask',
]
