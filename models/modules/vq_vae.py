from torch import nn
from models.modules.branch.audio import QuantizedAudio
from models.modules.branch.text import QuantizedText
from models.modules.codebook.EMA_codebook import VectorQuantizerEMA, MultiVectorQuantizerEMA
# from models.modules.codebook.gumbel_codebook import GumbelVectorQuantizer
from models.modules.codebook.kmeans_codebook import KmeansVectorQuantizer
# from models.modules.codebook.simple_codebook import CodeBook
from models.modules.codebook.simple_codebook import CodeBook


class SingleVQ(nn.Module):
    def __init__(self, dim, audio_dim, head_num, layer_num, code_size, groups, is_single, **kwargs):
        super().__init__()
        self.codebook = MultiVectorQuantizerEMA(code_size, dim, groups)
        self.audio_branch = QuantizedAudio(dim, audio_dim, head_num, layer_num, self.codebook, is_single=is_single)
        self.text_branch = QuantizedText(dim, head_num, layer_num, self.codebook, is_single=is_single)

    def project_text(self, text_data, text_mask, mask_ratio=0.0):
        return self.text_branch(text_data, text_mask, mask_ratio)

    def project_audio(self, audio_data, mask_ratio=0.0):
        return self.audio_branch(audio_data, mask_ratio)

    def cross_text_audio(self, audio_data, text_data, text_mask, mask_ratio=0.0):
        audio_core, audio_encoding, audio_extra_loss, audio_perplexity = self.audio_branch.quantize(audio_data)
        text_core, text_encoding, text_extra_loss, text_perplexity = self.text_branch.quantize(text_data, text_mask)
        audio_out, audio_mask = self.audio_branch.reconstruction(text_core, audio_data, mask_ratio)
        text_out, text_mask = self.text_branch.reconstruction(audio_core, text_data, text_mask, mask_ratio)
        return (audio_core, audio_encoding, audio_extra_loss, audio_out, audio_mask, audio_perplexity), \
               (text_core, text_encoding, text_extra_loss, text_out, text_mask, text_perplexity)

    def forward(self):
        pass

