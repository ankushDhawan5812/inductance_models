"""
Visualize model architecture summary and block diagram.

Usage:
    python visualize_arch.py configs/config_2d.yaml
    python visualize_arch.py configs/config_2d.yaml --output my_model
    python visualize_arch.py configs/config_2d.yaml --format svg
"""

import argparse
import os
from types import SimpleNamespace

import graphviz
import torch
import yaml
from torchinfo import summary

from model import (
    CausalPositionTransformer,
    PositionCNN,
    PositionCNN_AR,
    PositionCNN_LSTM,
    PositionLSTM,
    PositionTransformer,
)


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    flat = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
        else:
            flat.update(raw)
            break
    return SimpleNamespace(**flat)


def build_model(config):
    model_type = getattr(config, "model_type", "cnn")
    is_ar = model_type.endswith("_ar")
    output_dim = getattr(config, "output_dim", 3)
    window_size = config.window_size

    if model_type == "ar":
        model = PositionCNN_AR(
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            output_dim=output_dim,
        )
    elif model_type in ("lstm", "lstm_ar"):
        input_size = (1 + output_dim) if is_ar else 1
        model = PositionLSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            bidirectional=getattr(config, "bidirectional", False),
            output_dim=output_dim,
        )
    elif model_type in ("cnn_lstm", "cnn_lstm_ar"):
        input_size = (1 + output_dim) if is_ar else 1
        model = PositionCNN_LSTM(
            input_size=input_size,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            bidirectional=getattr(config, "bidirectional", False),
            output_dim=output_dim,
        )
    elif model_type in ("transformer", "transformer_ar"):
        input_size = (1 + output_dim) if is_ar else 1
        model = PositionTransformer(
            input_size=input_size,
            d_model=getattr(config, "d_model", 64),
            nhead=getattr(config, "nhead", 4),
            num_layers=getattr(config, "num_layers", 3),
            dim_feedforward=getattr(config, "dim_feedforward", 256),
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            max_len=window_size + 10,
            output_dim=output_dim,
        )
    elif model_type in ("causal_transformer", "causal_transformer_ar"):
        input_size = (1 + output_dim) if is_ar else 1
        model = CausalPositionTransformer(
            input_size=input_size,
            d_model=getattr(config, "d_model", 64),
            nhead=getattr(config, "nhead", 4),
            num_layers=getattr(config, "num_layers", 3),
            dim_feedforward=getattr(config, "dim_feedforward", 256),
            encoder_dims=getattr(config, "encoder_dims", [64]),
            decoder_dims=getattr(config, "decoder_dims", [128, 64]),
            dropout=config.dropout,
            max_len=window_size + 10,
            output_dim=output_dim,
        )
    else:
        model = PositionCNN(
            in_channels=1,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            output_dim=output_dim,
        )

    return model, model_type, is_ar, output_dim, window_size


def get_input_size(model_type, is_ar, output_dim, window_size):
    """Return the input_size tuple(s) for torchinfo.summary."""
    if model_type == "ar":
        return [(1, window_size, 1), (1, window_size, output_dim)]
    input_ch = (1 + output_dim) if is_ar else 1
    return (1, window_size, input_ch)


def build_block_diagram(model, model_type, input_shape_str, output_shape_str):
    """Build a clean module-level block diagram using graphviz."""
    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "TB",
            "fontname": "Helvetica",
            "bgcolor": "white",
            "pad": "0.5",
            "nodesep": "0.4",
            "ranksep": "0.5",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "shape": "record",
            "style": "filled,rounded",
            "fillcolor": "#f0f4ff",
            "color": "#4a6fa5",
            "penwidth": "1.5",
        },
        edge_attr={
            "color": "#666666",
            "arrowsize": "0.8",
        },
    )

    # Input node
    dot.node("input", f"Input\n{input_shape_str}", fillcolor="#e8f5e9", color="#4caf50")

    # Walk the module tree and create nodes for meaningful layers
    nodes = []
    for name, module in model.named_modules():
        if name == "":
            continue
        # Skip individual layers inside Sequential/TransformerEncoder to avoid clutter;
        # show only the top-level named children and their direct sub-modules
        depth = name.count(".")
        if depth > 1:
            continue

        class_name = module.__class__.__name__
        label = _module_label(name, module, class_name)
        if label is None:
            continue

        node_id = name.replace(".", "_")
        fillcolor = _color_for(class_name)
        dot.node(node_id, label, fillcolor=fillcolor)
        nodes.append(node_id)

    # Output node
    dot.node("output", f"Output\n{output_shape_str}", fillcolor="#fff3e0", color="#ff9800")

    # Chain edges sequentially
    all_nodes = ["input"] + nodes + ["output"]
    for a, b in zip(all_nodes, all_nodes[1:]):
        dot.edge(a, b)

    return dot


def _module_label(name, module, class_name):
    """Create a concise label for a module node."""
    if class_name == "Sequential":
        # Summarize contents
        children = list(module.children())
        child_types = [c.__class__.__name__ for c in children]
        summary_parts = []
        i = 0
        while i < len(child_types):
            ct = child_types[i]
            if ct == "Conv1d":
                c = children[i]
                summary_parts.append(f"Conv1d({c.in_channels}→{c.out_channels}, k={c.kernel_size[0]})")
            elif ct == "Linear":
                c = children[i]
                summary_parts.append(f"Linear({c.in_features}→{c.out_features})")
            elif ct in ("BatchNorm1d", "ReLU", "Dropout", "GELU"):
                # Group activation/norm with previous
                if summary_parts:
                    extras = [ct]
                    j = i + 1
                    while j < len(child_types) and child_types[j] in ("BatchNorm1d", "ReLU", "Dropout", "GELU", "LayerNorm"):
                        extras.append(child_types[j])
                        j += 1
                    summary_parts[-1] += " + " + ", ".join(extras)
                    i = j
                    continue
                else:
                    summary_parts.append(ct)
            else:
                summary_parts.append(ct)
            i += 1
        label = f"{name}\n" + "\n".join(summary_parts)
        return label
    elif class_name == "LSTM":
        return (
            f"{name} (LSTM)\n"
            f"input={module.input_size}, hidden={module.hidden_size}\n"
            f"layers={module.num_layers}, bidir={module.bidirectional}"
        )
    elif class_name == "TransformerEncoder":
        layer = module.layers[0]
        return (
            f"{name} (TransformerEncoder)\n"
            f"{module.num_layers} layers\n"
            f"d_model={layer.self_attn.embed_dim}, heads={layer.self_attn.num_heads}\n"
            f"ff_dim={layer.linear1.in_features}→{layer.linear1.out_features}"
        )
    elif class_name == "Linear":
        return f"{name} (Linear)\n{module.in_features}→{module.out_features}"
    elif class_name in ("Dropout", "AdaptiveAvgPool1d"):
        return f"{name}\n{class_name}"
    elif class_name == "PositionalEncoding":
        return f"{name}\nPositionalEncoding"
    # Skip containers and non-informative wrappers
    return None


def _color_for(class_name):
    """Return a fill color based on layer type."""
    colors = {
        "Sequential": "#e3f2fd",
        "LSTM": "#f3e5f5",
        "TransformerEncoder": "#f3e5f5",
        "Linear": "#fff9c4",
        "PositionalEncoding": "#e0f7fa",
        "Dropout": "#fce4ec",
        "AdaptiveAvgPool1d": "#e8eaf6",
    }
    return colors.get(class_name, "#f0f4ff")


def main():
    parser = argparse.ArgumentParser(description="Visualize model architecture")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--output", "-o", default=None, help="Output filename (without extension)")
    parser.add_argument("--format", "-f", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    config = load_config(args.config)
    model, model_type, is_ar, output_dim, window_size = build_model(config)
    model.eval()

    out_dir = "model_archs"
    os.makedirs(out_dir, exist_ok=True)
    base_name = args.output or f"arch_{model_type}"

    # 1) Print torchinfo summary and save to text file
    input_size = get_input_size(model_type, is_ar, output_dim, window_size)
    model_summary = summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        verbose=0,
    )
    summary_path = os.path.join(out_dir, f"{base_name}.txt")
    with open(summary_path, "w") as f:
        f.write(str(model_summary))
    print(str(model_summary))
    print(f"\nSummary saved to: {summary_path}")

    # 2) Build and render block diagram
    if model_type == "ar":
        input_shape_str = f"(B, {window_size}, 1) + (B, {window_size}, {output_dim})"
    else:
        input_ch = (1 + output_dim) if is_ar else 1
        input_shape_str = f"(B, {window_size}, {input_ch})"
    output_shape_str = f"(B, {output_dim})"

    dot = build_block_diagram(model, model_type, input_shape_str, output_shape_str)
    dot.format = args.format
    diagram_path = dot.render(os.path.join(out_dir, base_name), cleanup=True)
    print(f"Diagram saved to: {diagram_path}")


if __name__ == "__main__":
    main()
