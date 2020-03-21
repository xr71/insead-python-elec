from graphviz import Digraph
import tensorflow as tf

def get_neuron_name(layer_name, neuron_idx):
    return layer_name + "_" + str(neuron_idx)

def activation_to_str(act):
    if act.__name__ != "linear":
        return act.__name__

def draw_nn_graph(m):
    # initialize graph
    g = Digraph()
    g.graph_attr['rankdir'] = 'LR'
    n_inputs = m.input_shape[1]
    for input_idx in range(n_inputs):
        input_name = 'input_' + str(input_idx)
        g.node(input_name, shape="circle")

    # book-keeping arrays; slightly more complex to correctly treat 'input'
    layer_names = ['input']
    layer_sizes = [n_inputs]
    w = m.get_weights()
    for idx, layer in enumerate(m.layers):
        layer_names.append(layer.name)
        layer_sizes.append(w[2*idx].shape[1])

    # starts at first hidden layer
    for layer_idx, layer in enumerate(m.layers):
        w_layer = w[2*layer_idx]
        b_layer = w[2*layer_idx + 1]
        for neuron_idx, neuron_inbound_weights in enumerate(w_layer.T):
            name = get_neuron_name(layer.name, neuron_idx)
            neuron_label = name + "\n" + f"bias: {b_layer[neuron_idx]:.2f}"
            act_label = activation_to_str(layer.activation)
            if act_label is not None:
                neuron_label = neuron_label + "\n" + f"act: {act_label}"
            g.node(name, neuron_label, shape="circle")
            # note the implicit shift back one even though we just write `idx`
            for prev_neuron_idx in range(layer_sizes[layer_idx]):
                prev_name = get_neuron_name(
                    layer_names[layer_idx],
                    prev_neuron_idx
                )
                edge_label = str(round(neuron_inbound_weights[prev_neuron_idx],2))
                g.edge(prev_name, name, label=edge_label)

    return(g)

