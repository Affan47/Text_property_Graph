"""Windowed, source-aligned embeddings shared by general and security TPGs."""
from tpg.schema.types import NodeType


def encode_tokens(tokenizer, model, text, device, batch_size=16):
    import torch
    limit = min(512, getattr(model.config, "max_position_embeddings", 512))
    encoded = tokenizer(text, truncation=True, max_length=limit, padding=True,
        return_offsets_mapping=True, return_overflowing_tokens=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping", None)
    vectors, positions = [], []
    for start in range(0, encoded["input_ids"].size(0), batch_size):
        inputs = {key: value[start:start + batch_size].to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state.cpu()
        off = offsets[start:start + batch_size]
        mask = off[..., 1] > off[..., 0]
        vectors.append(hidden[mask])
        positions.append(off[mask])
    return torch.cat(vectors), torch.cat(positions)


def encode_texts(tokenizer, model, texts, device, batch_size=16):
    import torch
    result = []
    for text in texts:
        vectors, _ = encode_tokens(tokenizer, model, text, device, batch_size)
        result.append(vectors.mean(dim=0) if len(vectors) else torch.zeros(model.config.hidden_size))
    return torch.stack(result) if result else torch.empty((0, model.config.hidden_size))


def embed_graph(graph, tokenizer, model, device):
    """Pool sentence windows and align every textual span to contextual tokens."""
    import torch
    for sentence in graph.nodes(NodeType.SENTENCE):
        vectors, offsets = encode_tokens(tokenizer, model, sentence.properties.text, device)
        if not len(vectors):
            continue
        sentence.properties.extra["embedding"] = vectors.mean(dim=0).tolist()
        for node in graph.nodes():
            if node.node_type in (NodeType.DOCUMENT, NodeType.PARAGRAPH, NodeType.SENTENCE, NodeType.TOPIC):
                continue
            p, sp = node.properties, sentence.properties
            if not sp.char_start <= p.char_start < p.char_end <= sp.char_end:
                continue
            start, end = p.char_start - sp.char_start, p.char_end - sp.char_start
            mask = (offsets[:, 0] < end) & (offsets[:, 1] > start)
            if torch.any(mask):
                p.extra["embedding"] = vectors[mask].mean(dim=0).tolist()
    graph.metadata.update(embedding_dim=model.config.hidden_size,
        transformer_model=getattr(model.config, "_name_or_path", "unknown"),
        transformer_revision=getattr(model.config, "_commit_hash", None),
        embedding_method="contextual_mean_nonoverlapping_windows",
        embedded_nodes=sum("embedding" in n.properties.extra for n in graph.nodes()))


class EmbeddingPass:
    name = "embedding_pass"

    def __init__(self, model_name, device="cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def run(self, graph):
        if self.name not in graph.passes_applied:
            embed_graph(graph, self.tokenizer, self.model, self.device)
            graph.mark_pass(self.name)
        return graph
