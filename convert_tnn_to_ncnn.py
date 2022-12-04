from collections import OrderedDict
import struct


CONVERT_FUNC = {}
def register(type):
    def wrapper(func):
        def decorated(*args, **kwargs):
            return func(*args, **kwargs)

        CONVERT_FUNC[type] = decorated
        return decorated

    return wrapper


def correspond_param_convert(in_params, correspond_dict):
    out_params = ""
    for key in correspond_dict.keys():
        index = correspond_dict[key]
        if isinstance(index, str):
            out_params += str(key) + "=" + index + " "
            continue

        if index < len(in_params) and index >= 0:
            out_params += str(key) + "=" + in_params[index] + " "

    return out_params


@register("Pad")
def Pad_convert(in_params):
    '''
    tnn: n1 n2 pad_t pad_b pad_l pad_r pad_c_b pad_c_e type value
    ncnn: top bottom left right type value per_channel_pad_data_size front behind
    '''

    correspond_dict = OrderedDict({0: 2, 1: 3, 2: 4, 3: 5, 4: 8, 5: 9, 7: 6, 8: 7})
    out_params = correspond_param_convert(in_params, correspond_dict)

    out_dict = {}
    out_dict["layer_type"] = "Padding"
    return [out_params, out_dict]


@register("Convolution")
def Convolution_convert(in_params, **kwargs):
    '''
    tnn: group input_channel output_channel kernel_h kernel_w stride_h stride_w pad_h pad_w bias pad_type
            dialation_h dialation_w activation_type
    ncnn: num_output kernel_w dilation_w stride_w pad_left bias_term weight_data_size 7
            int8_scale_term activation_type activation_params kernel_h dilation_h stride_h
            pad_top pad_right pad_bottom 17 pad_value dynamic_weight
    '''
    correspond_dict = OrderedDict({0: 2, 1: 4, 2: 12, 3: 6, 4: 8, 5: 9, 9: 13,
                                   11: 3, 12: 11, 13: 5, 14: 7})

    weight_data_size = kwargs["weight_data_size"]
    correspond_dict[6] = str(weight_data_size)
    layer_type = "Convolution"
    if int(in_params[0]) > 1:
        layer_type = "ConvolutionDepthWise"
        correspond_dict[7] = in_params[2]

    out_params = correspond_param_convert(in_params, correspond_dict)

    out_dict = {}
    out_dict["layer_type"] = layer_type
    out_dict["output_channel"] = int(in_params[2])
    return [out_params, out_dict]


@register("ReLU")
def ReLU_convert(in_params):
    return ""


@register("Add")
def Add_convert(in_params):
    out_params = "0=0"

    out_dict = {}
    out_dict["layer_type"] = "BinaryOp"
    return [out_params, out_dict]


@register("Pooling")
def Pooling_convert(in_params):
    '''
    tnn: pool_type kernel_h kernel_w stride_h stride_w pad_h pad_w kernel_index_h kernel_index_w
            pad_type ceil_mode is_adaptive_pool output_h output_w
    ncnn: pooling_type kernel_w stride_w pad_left global_pooling pad_mode avgpool_count_include_pad
            adaptive_pooling out_w 9 10 kernel_h stride_h pad_top pad_right pad_bottom 16 17 out_h
    '''
    correspond_dict = OrderedDict({0: 0, 1: 2, 2: 4, 3: 6, 7: 11, 8: 13, 11: 1, 12: 3, 13: 5, 18: 12})
    global_pooling = 0
    if int(in_params[1]) == 0 and int(in_params[2]) == 0:
        global_pooling = 1

    correspond_dict[4] = str(global_pooling)
    correspond_dict[5] = in_params[9]
    if int(in_params[9]) < 0:
        correspond_dict[5] = str(0)

    return correspond_param_convert(in_params, correspond_dict)


@register("Permute")
def Permute_convert(in_params):
    order_size = int(in_params[0])
    order_type = None
    if order_size == 4:
        i0, i1, i2 = in_params[2], in_params[3], in_params[4]
        if i0 == "2" and i1 == "3" and i2 == "1":
            order_type = 3

    out_params = ""
    if order_type is not None:
        out_params = "0=" + str(order_type)
    return out_params


@register("Reshape")
def Reshape_convert(in_params):
    '''
    tnn: axis num_axes top_blob_dim_size shape reshape_type
    ncnn: w h c permute 4 5 6 7 8 9 10 d
    '''
    top_blob_dim_size = int(in_params[2])
    if top_blob_dim_size == 4:
        correspond_dict = OrderedDict({0: 6, 1: 5, 2: 4})
        return correspond_param_convert(in_params, correspond_dict)
    elif top_blob_dim_size == 3:
        correspond_dict = OrderedDict({0: 5, 1: 4})
        return correspond_param_convert(in_params, correspond_dict)


@register("Concat")
def Concat_convert(in_params):
    '''
    tnn: axis
    ncnn: axis
    '''
    correspond_dict = OrderedDict({})
    correspond_dict[0] = str(int(in_params[0]) - 1) #tnn的维度从d开始算，在3维的情况下，ncnn的维度从c开始算
    return correspond_param_convert(in_params, correspond_dict)


@register("Clip")
def Clip_convert(in_params):
    '''
    tnn: min max
    ncnn: min max
    '''
    correspond_dict = OrderedDict({0: 0, 1: 1})
    return correspond_param_convert(in_params, correspond_dict)


@register("SoftmaxCaffe")
def Softmax_convert(in_params):
    '''
    tnn: axis
    ncnn: axis fixbug0
    '''
    correspond_dict = OrderedDict({0: 0})
    correspond_dict[1] = "1"
    out_params = correspond_param_convert(in_params, correspond_dict)

    out_dict = {}
    out_dict["layer_type"] = "Softmax"
    return [out_params, out_dict]


@register("Sigmoid")
def Sigmoid_convert(in_params):
    return ""


@register("InnerProduct")
def InnerProduct_convert(in_params, **kwargs):
    '''
    tnn: num_output has_bias transpose axis
    ncnn: num_output bias_term weight_data_size int8_scale_term activation_type activation_params
    '''
    correspond_dict = OrderedDict({0: 0, 1: 1})
    weight_data_size = kwargs["weight_data_size"]
    correspond_dict[2] = str(weight_data_size)#str(int(in_params[0]) * input_channel)

    return correspond_param_convert(in_params, correspond_dict)


@register("PReLU")
def PReLU_convert(in_params, **kwargs):
    '''
    tnn: channel_shared has_filler
    ncnn: num_slope
    '''
    weight_data_size = kwargs["weight_data_size"]
    if int(in_params[0]) == 1:
        out_params = "0=1"
    else:
        out_params = "0=" + str(weight_data_size)
    return out_params


@register("Upsample")
def Upsample_convert(in_params):
    '''
    tnn: mode scale_h scale_w align_corners
    ncnn: resize_type height_scale width_scale output_height output_width dynamic_target_size align_corner
    '''

    correspond_dict = OrderedDict({0: 0, 6: 3})
    out_params = correspond_param_convert(in_params, correspond_dict)
    out_params += " 1=" + '{:e}'.format(float(in_params[1]))
    out_params += " 2=" + '{:e}'.format(float(in_params[2]))

    out_dict = {}
    out_dict["layer_type"] = "Interp"
    return [out_params, out_dict]


@register("SplitV")
def SplitV_convert(in_params):
    '''
    tnn: axis slice_count slice1 ...
    ncnn: slices axis
    '''
    out_params = "1=" + str(int(in_params[0]) - 1) + " -23300="
    slice_count = int(in_params[1])
    out_params += str(slice_count)
    for i in range(slice_count - 1):
        out_params += "," + in_params[i + 2]

    out_params += ",-233"

    out_dict = {}
    out_dict["layer_type"] = "Slice"

    return [out_params, out_dict]


@register("ShuffleChannel")
def ShuffleChannel_convert(in_params):
    '''
    tnn: group
    ncnn: group reverse
    '''
    correspond_dict = OrderedDict({0: 0})
    return correspond_param_convert(in_params, correspond_dict)


FIRST_PARAM_SPACE = 17
SECOND_THIRD_INTERVAL = 19


def convert_param(tnn_file, ncnn_file, weight_data_size_dict=None):
    with open(tnn_file) as tnn_proto:
        tnn_layers = tnn_proto.readlines()

    inputs = tnn_layers[1]
    inputname = inputs.strip().replace('"', "").split()[0]
    outputs = tnn_layers[3]

    ncnn_layers = []
    ncnn_layers.append("7767517\n")

    ncnn_inputs = "Input" + " " * (FIRST_PARAM_SPACE - len("Input")) + inputname
    ncnn_inputs += " " * SECOND_THIRD_INTERVAL + "0 1 " + inputname + "\n"
    ncnn_layers.append(ncnn_inputs)

    for i in range(5, len(tnn_layers)):
        tnn_layer = tnn_layers[i]
        tnn_layer = tnn_layer.strip().replace('"', "").replace(",", "").split()
        layer_type = tnn_layer[0]
        layer_name = tnn_layer[1]
        input_num = int(tnn_layer[2])
        output_num = int(tnn_layer[3])

        input_names = []
        for j in range(4, 4 + input_num):
            input_names.append(tnn_layer[j])

        output_names = []
        for j in range(4 + input_num, 4 + input_num + output_num):
            output_names.append(tnn_layer[j])

        layer_param = tnn_layer[4 + input_num + output_num:]

        if layer_type not in CONVERT_FUNC.keys():
            print("not support layer type: ", layer_type)
            continue

        if layer_type == "Convolution" or layer_type == "InnerProduct" or layer_type == "PReLU":
            if layer_name not in weight_data_size_dict.keys():
                print("weight_data_size is not found")

            weight_data_size = weight_data_size_dict[layer_name]
            output = CONVERT_FUNC[layer_type](layer_param, weight_data_size=weight_data_size)
        else:
            output = CONVERT_FUNC[layer_type](layer_param)
        if output is None:
            continue

        if isinstance(output, list):
            out_param, out_dict = output
            if "output_channel" in out_dict.keys():
                input_channel = out_dict["output_channel"]

            if "layer_type" in out_dict.keys():
                layer_type = out_dict["layer_type"]
        else:
            out_param = output

        ncnn_layer = layer_type
        if len(ncnn_layer) < FIRST_PARAM_SPACE:
            ncnn_layer += " " * (FIRST_PARAM_SPACE - len(ncnn_layer))
        else:
            ncnn_layer += " "

        ncnn_layer += layer_name + " " * SECOND_THIRD_INTERVAL
        ncnn_layer += str(input_num) + " " + str(output_num)
        for n in input_names:
            ncnn_layer += " " + n

        for n in output_names:
            ncnn_layer += " " + n

        #print(layer_name, "out_param", out_param)
        if len(out_param) > 0:
            ncnn_layer += " " + out_param

        ncnn_layer = ncnn_layer.rstrip()
        ncnn_layer += "\n"

        ncnn_layers.append(ncnn_layer)

    layer_count = len(ncnn_layers) - 1


    def get_blob_count(ncnn_layers):
        blobs = set()
        for i in range(1, len(ncnn_layers)):
            layer = ncnn_layers[i]
            layer = layer.strip().split()
            inputnum = int(layer[2])
            outputnum = int(layer[3])
            blobnum = inputnum + outputnum

            for j in range(4, 4 + blobnum):
                blobs.add(layer[j])

        return len(blobs)

    blob_count = get_blob_count(ncnn_layers)

    ncnn_layers.insert(1, str(layer_count) + " " + str(blob_count) + "\n")

    with open(ncnn_file, "w") as ncnn_param:
        for layer in ncnn_layers:
            ncnn_param.write(layer)


RESOURCE_FUNC = {}
def register_res(type):
    def wrapper(func):
        def decorated(*args, **kwargs):
            return func(*args, **kwargs)

        RESOURCE_FUNC[type] = decorated
        return decorated

    return wrapper


def bytes_to_str(bytearr):
    bytearr = [b.decode("utf-8") for b in bytearr]
    string = "".join(bytearr)
    return string


def get_int(fmodel):
    buff = fmodel.read(4)
    return struct.unpack("i", buff)[0]


def get_uint(fmodel):
    buff = fmodel.read(4)
    return struct.unpack("I", buff)[0]


def get_str(fmodel, strlen):
    buff = fmodel.read(strlen)
    string = struct.unpack("c" * strlen, buff)
    return bytes_to_str(string)


def get_raw(fmodel):
    magic_number = get_uint(fmodel)
    data_type = get_int(fmodel)
    length = get_int(fmodel)

    if 4206624772 == magic_number:
        size = get_int(fmodel)
        for i in range(size):
            dim = get_int(fmodel)

    buffer = fmodel.read(length)
    return buffer


@register_res("Convolution")
def Convolution_res_convert(fmodel):
    layer_name_len = get_int(fmodel)
    layer_name = get_str(fmodel, layer_name_len)
    has_bias = get_int(fmodel)

    filter_buffer = get_raw(fmodel)
    filter_len = len(filter_buffer) // 4

    flag_struct = 0
    flag_struct_buffer = struct.pack("I", flag_struct)
    filter_buffer = flag_struct_buffer + filter_buffer

    bias_buffer = ""
    if has_bias:
        bias_buffer = get_raw(fmodel)

    return (filter_buffer + bias_buffer, filter_len)


@register_res("InnerProduct")
def InnerProduct_res_convert(fmodel):
    layer_name_len = get_int(fmodel)
    layer_name = get_str(fmodel, layer_name_len)

    filter_buffer = get_raw(fmodel)
    filter_len = len(filter_buffer) // 4

    flag_struct = 0
    flag_struct_buffer = struct.pack("I", flag_struct)
    filter_buffer = flag_struct_buffer + filter_buffer

    bias_buffer = get_raw(fmodel)

    return (filter_buffer + bias_buffer, filter_len)


@register_res("PReLU")
def PReLU_res_convert(fmodel):
    layer_name_len = get_int(fmodel)
    layer_name = get_str(fmodel, layer_name_len)

    slope_buffer = get_raw(fmodel)
    slope_len = len(slope_buffer) // 4

    flag_struct = 0
    flag_struct_buffer = struct.pack("I", flag_struct)
    slope_buffer = flag_struct_buffer + slope_buffer

    return (slope_buffer, slope_len)


def convert_model(tnn_model, ncnn_model, **kwargs):
    weight_data_size_dict = {}
    with open(ncnn_model, "wb") as fncnn:
        with open(tnn_model, "rb") as fmodel:
            magic_version_number = get_uint(fmodel)
            layer_cnt = get_int(fmodel)

            for i in range(layer_cnt):
                layer_type = get_int(fmodel)

                type_str_len = get_int(fmodel)
                type_str = get_str(fmodel, type_str_len)

                name_len = get_int(fmodel)
                name = get_str(fmodel, name_len)

                if type_str not in RESOURCE_FUNC.keys():
                    print("Not supported", type_str)
                    continue

                buffer = RESOURCE_FUNC[type_str](fmodel)

                if "filtered_layers" in kwargs.keys():
                    if name in kwargs["filtered_layers"]:
                        print("filtered ", name)
                        continue

                if isinstance(buffer, tuple):
                    buffer, weight_data_size = buffer
                    weight_data_size_dict[name] = weight_data_size

                fncnn.write(buffer)
    return weight_data_size_dict

if __name__ == "__main__":
    weight_data_size_dict = convert_model(
        r"D:\Program\Project\ncnn-20210322\examples\ncnn_examples\TNN_face_alignment\youtu_face_alignment_phase1.tnnmodel",
        r"youtu_face_alignment_phase1_new.bin", filtered_layers=("853",))

    convert_param(r"D:\Program\Project\ncnn-20210322\examples\ncnn_examples\TNN_face_alignment\youtu_face_alignment_phase1.tnnproto",
        r"youtu_face_alignment_phase1_new.param", weight_data_size_dict)


    # convert_param(
    #     r"D:\Program\Project\ncnn-20210322\examples\ncnn_examples\TNN_face_alignment\youtu_face_alignment_phase2.tnnproto",
    #     r"D:\Program\Project\ncnn-20220420\examples\youtu_face_alignment_phase2.param",
    #     r"D:\Program\Project\ncnn-20210322\examples\ncnn_examples\TNN_face_alignment\youtu_face_alignment_phase2_output_shape.txt")
    #
    # convert_model(
    #     r"D:\Program\Project\ncnn-20210322\examples\ncnn_examples\TNN_face_alignment\youtu_face_alignment_phase2.tnnmodel",
    #     r"D:\Program\Project\ncnn-20220420\examples\youtu_face_alignment_phase2.bin")

    """
    添加一行 Permute          351                   1 1 374 375 0=1
    """
    # convert_param(
    #     r"D:\Program\Project\ncnn-20210322\examples\ncnn_examples\TNN_Face_Detection\version-slim-320_simplified.tnnproto",
    #     r"D:\Program\Project\ncnn-20220420\examples\version-slim-320_simplified.param",
    #     r"D:\Program\Project\ncnn-20210322\examples\ncnn_examples\TNN_Face_Detection\version-slim-320_simplified_output_shape.txt")
    #
    # convert_model(
    #     r"D:\Program\Project\ncnn-20210322\examples\ncnn_examples\TNN_Face_Detection\version-slim-320_simplified.tnnmodel",
    #     r"D:\Program\Project\ncnn-20220420\examples\version-slim-320_simplified.bin")

    # convert_param(
    #     r"D:\Program\Project\TNN-master\examples\tnn-models-master\model\nanodet\nanodet_m.tnnproto",
    #     r"D:\Program\Project\TNN-master\examples\tnn-models-master\model\nanodet\nanodet_m.param",
    #     r"D:\Program\Project\TNN-master\examples\tnn-models-master\model\nanodet\nanodet_m_output_shape.txt", input_channel=3)
    #
    # convert_model(
    #     r"D:\Program\Project\TNN-master\examples\tnn-models-master\model\nanodet\nanodet_m.tnnmodel",
    #     r"D:\Program\Project\TNN-master\examples\tnn-models-master\model\nanodet\nanodet_m.bin")

