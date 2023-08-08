import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import cluster
from types import MethodType
import torch
import diffusers


def plot_svd_images(feats, save_name):
    u, s, v = torch.linalg.svd(feats.float())
    dim = int(u.shape[-1] ** 0.5)
    u = u.reshape(dim, dim, -1).numpy()
    full_img = np.zeros((u.shape[-1], u.shape[-1], 3))
    for i in range(u.shape[-1]):
        col = (i % dim) * dim
        row = (i // dim) * dim
        plt.imsave("temp.png", u[:, :, i])
        image = Image.open("temp.png").convert("RGB")
        np_img = np.array(image)
        full_img[row:row + dim, col:col + dim, :] = np_img
    plots = Image.fromarray(full_img.astype(np.uint8))
    plots.save(f"{save_name}")


def plot_feats(feats, save_name):
    dim = int(feats.shape[0] ** 0.5)
    feats = feats.reshape(dim, dim, -1)
    full_img = np.zeros((feats.shape[0] ** 2, feats.shape[0] ** 2, 3))
    for i in range(feats.shape[-1]):
        col = (i % dim) * dim
        row = (i // dim) * dim
        plt.imsave("temp.png", feats[:, :, i])
        image = Image.open("temp.png").convert("RGB")
        np_img = np.array(image)
        full_img[row:row + dim, col:col + dim, :] = np_img
    plots = Image.fromarray(full_img.astype(np.uint8))
    plots.save(f"{save_name}")


colors = [
    np.array([255, 0, 0]),
    np.array([0, 255, 0]),
    np.array([0, 0, 255]),
    np.array([255, 255, 0]),
    np.array([0, 255, 255]),
    np.array([255, 0, 255]),
    np.array([127, 0, 255]),
    np.array([128, 128, 128]),
]


def kmeans_cluster(feats, save_name, n_clusters=4):
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(feats)
    dim = int(feats.shape[0] ** 0.5)
    marked_image = np.zeros((dim, dim, 3))
    for i in range(feats.shape[0]):
        cluster_num = kmeans.predict(feats[i:i + 1]).item()
        color = colors[cluster_num]
        col = i % dim
        row = i // dim
        marked_image[row, col, :] = color

    marked_image = Image.fromarray(marked_image.astype(np.uint8))
    marked_image.save(f"{save_name}")

    return marked_image


def run_tests(model_path, prompt="cinematic shot of a woman at a desk looking out a window hd, melancholy",
              sd_params={"height": 512, "width": 512},
              seed=124,
              # feats=["query","key","incoming","self_maps"],
              # steps_to_check=[5,20,50],
              # modes=["uncond","cond"],
              # sides=['down','up'],

              feats=["incoming", "query"],
              steps_to_check=[5, 20, 50],
              modes=["cond"],
              sides=['up', 'down'],
              n_clusters=5,
              block_num=1,
              attn_num=-1,
              transformer_num=0  # 1.5/2.0 only have 1 layer anyway
              ):
    # warning: doing svd on the higher resolution blocks will eat up all your vram

    modes_dict = {"uncond": 0,
                  "cond": 1}

    class Store:
        def __init__(self):
            self.cur_timestep = None
            self.down_feature_store = {}
            self.up_feature_store = {}

    the_store = Store()

    def unet_time_hook(module, input):
        the_store.cur_timestep = int((1 - input[1].item() / 1000) * 50)

    def make_hook(module, feat="query", place="down"):

        def attention_map_hook(*inputs):
            # remove self
            inputs = inputs[1:]
            # now here we will have 8 heads
            output = module.old_get_attention_scores(*inputs)
            output_to_save = output.clone().cpu()
            uncond, cond = output_to_save.chunk(2)
            output_to_save = torch.cat([uncond.mean(0, keepdim=True), cond.mean(0, keepdim=True)])
            if "down" in place:
                if the_store.down_feature_store.get(feat, None) is None:
                    the_store.down_feature_store[feat] = {}
                the_store.down_feature_store[feat][the_store.cur_timestep] = output_to_save
            elif "up" in place:
                if the_store.up_feature_store.get(feat, None) is None:
                    the_store.up_feature_store[feat] = {}
                the_store.up_feature_store[feat][the_store.cur_timestep] = output_to_save

            return output

        def feat_hook(module, input, output):
            if "down" in place:
                if the_store.down_feature_store.get(feat, None) is None:
                    the_store.down_feature_store[feat] = {}
                the_store.down_feature_store[feat][the_store.cur_timestep] = output.clone().cpu()
            elif "up" in place:
                if the_store.up_feature_store.get(feat, None) is None:
                    the_store.up_feature_store[feat] = {}
                the_store.up_feature_store[feat][the_store.cur_timestep] = output.clone().cpu()

        def incoming_feat_hook(module, input):
            if "down" in place:
                if the_store.down_feature_store.get(feat, None) is None:
                    the_store.down_feature_store[feat] = {}
                the_store.down_feature_store[feat][the_store.cur_timestep] = input[0].clone().cpu()
            elif "up" in place:
                if the_store.up_feature_store.get(feat, None) is None:
                    the_store.up_feature_store[feat] = {}
                the_store.up_feature_store[feat][the_store.cur_timestep] = input[0].clone().cpu()

        if feat in ["incoming"]:
            module.register_forward_pre_hook(incoming_feat_hook)
        elif feat in ["query", "key"]:
            module.register_forward_hook(feat_hook)
        elif feat in ['self_maps', "cross_maps"]:
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = MethodType(attention_map_hook, module)

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_path).to("cuda", torch.float16)
    pipe.unet.register_forward_pre_hook(unet_time_hook)
    torch.manual_seed(seed)

    for feat in feats:
        if "query" in feat:
            make_hook(pipe.unet.down_blocks[block_num].attentions[attn_num].transformer_blocks[0].attn1.to_q, feat=feat,
                      place="down")
            make_hook(pipe.unet.up_blocks[3 - block_num].attentions[attn_num].transformer_blocks[0].attn1.to_q,
                      feat=feat, place="up")
        if "key" in feat:
            make_hook(pipe.unet.down_blocks[block_num].attentions[attn_num].transformer_blocks[0].attn1.to_k, feat=feat,
                      place="down")
            make_hook(pipe.unet.up_blocks[3 - block_num].attentions[attn_num].transformer_blocks[0].attn1.to_k,
                      feat=feat, place="up")
        if "incoming" in feat:
            make_hook(pipe.unet.down_blocks[block_num].attentions[attn_num].transformer_blocks[0].attn1.to_out[0],
                      feat=feat, place="down")
            make_hook(pipe.unet.up_blocks[3 - block_num].attentions[attn_num].transformer_blocks[0].attn1.to_out[0],
                      feat=feat, place="up")
        if "self_maps" in feat:
            make_hook(pipe.unet.down_blocks[block_num].attentions[attn_num].transformer_blocks[0].attn1, feat=feat,
                      place="down")
            make_hook(pipe.unet.up_blocks[3 - block_num].attentions[attn_num].transformer_blocks[0].attn1, feat=feat,
                      place="up")
        if "cross_maps" in feat:
            make_hook(pipe.unet.down_blocks[block_num].attentions[attn_num].transformer_blocks[0].attn2, feat=feat,
                      place="down")
            make_hook(pipe.unet.up_blocks[3 - block_num].attentions[attn_num].transformer_blocks[0].attn2, feat=feat,
                      place="up")

    img = pipe(prompt, height=768, width=768).images
    for feat in feats:
        for step in steps_to_check:
            for mode in modes:
                for side in sides:
                    idx = modes_dict[mode]
                    if "down" in side:
                        feature = the_store.down_feature_store[feat][step - 1][idx]
                    else:
                        feature = the_store.up_feature_store[feat][step - 1][idx]

                    filename = f"{feat}_{side}_{mode}_{step}.png"
                    plot_feats(feature, filename)

                    filename = f"svd_{feat}_{side}_{mode}_{step}.png"
                    plot_svd_images(feature, filename)

                    filename = f"segment_{feat}_{side}_{mode}_{step}.png"
                    kmeans_cluster(feature, filename, n_clusters=n_clusters)






