import torch
from torch.utils.data import DataLoader
import pandas as pd
import tqdm

from book import Goodbooks, df

hidden_dim = 16
# 建立训练和验证dataloader
traindataset = Goodbooks(df, 'training')
validdataset = Goodbooks(df, 'validation')
# 确保您已经定义了NCFModel类

# 构建模型
class NCFModel(torch.nn.Module):
    def __init__(self, hidden_dim, user_num, item_num, mlp_layer_num=4, weight_decay=1e-5, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_num = user_num
        self.item_num = item_num
        self.mlp_layer_num = mlp_layer_num
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.mlp_user_embedding = torch.nn.Embedding(user_num, hidden_dim * (2 ** (self.mlp_layer_num - 1)))
        self.mlp_item_embedding = torch.nn.Embedding(item_num, hidden_dim * (2 ** (self.mlp_layer_num - 1)))

        self.gmf_user_embedding = torch.nn.Embedding(user_num, hidden_dim)
        self.gmf_item_embedding = torch.nn.Embedding(item_num, hidden_dim)

        mlp_Layers = []
        input_size = int(hidden_dim * (2 ** (self.mlp_layer_num)))
        for i in range(self.mlp_layer_num):
            mlp_Layers.append(torch.nn.Linear(int(input_size), int(input_size / 2)))
            mlp_Layers.append(torch.nn.Dropout(self.dropout))
            mlp_Layers.append(torch.nn.ReLU())
            input_size /= 2
        self.mlp_layers = torch.nn.Sequential(*mlp_Layers)

        self.output_layer = torch.nn.Linear(2 * self.hidden_dim, 1)

    def forward(self, user, item):
        user_gmf_embedding = self.gmf_user_embedding(user)
        item_gmf_embedding = self.gmf_item_embedding(item)

        user_mlp_embedding = self.mlp_user_embedding(user)
        item_mlp_embedding = self.mlp_item_embedding(item)

        gmf_output = user_gmf_embedding * item_gmf_embedding

        mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        output = torch.sigmoid(self.output_layer(torch.cat([gmf_output, mlp_output], dim=-1))).squeeze(-1)

        # return -r_pos_neg + reg
        return output
# 加载模型权重
model_path = 'model_epoch_2.h5'  # 假设这是您最后一次保存的模型文件
model = NCFModel(hidden_dim, traindataset.user_nums, traindataset.book_nums)  # 创建一个新的模型实例
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 加载权重
model.eval()  # 设置模型为评估模式

# 如果您在GPU上训练模型，但现在想在CPU上运行预测，确保模型在CPU上
model.to(torch.device('cpu'))

# 读取测试数据集
df_test = pd.read_csv('test_dataset.csv')
user_for_test = df_test['user_id'].tolist()


# 预测
def predict_for_users(model, users, item_nums):
    predict_item_ids = []
    with torch.no_grad():
        for user in tqdm.tqdm(users):
            # 获取用户未交互过的物品
            user_visited_items = traindataset.user_book_map[user]
            items_for_predict = list(set(range(item_nums)) - set(user_visited_items))

            # 将物品ID转换为张量
            items_tensor = torch.LongTensor(items_for_predict).to(model.device)

            # 获取预测分数
            user_tensor = torch.LongTensor([user]).to(model.device)
            scores = model.predict(user_tensor, items_tensor)

            # 获取前10个最高分对应的物品ID
            top_items = (-scores).argsort()[:10]
            predict_item_ids.append(top_items.cpu().numpy())
    return predict_item_ids


# 进行预测
predictions = predict_for_users(model, user_for_test, traindataset.book_nums)

# 将预测结果写入CSV文件
with open('submission.csv', 'w', encoding='utf-8') as f:
    for user, item_ids in zip(user_for_test, predictions):
        for item_id in item_ids:
            f.write(f'{user},{item_id}\n')

print("预测完成，结果已写入submission_using_model2.csv文件。")
