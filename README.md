# 依赖
python3.9
# 安装/升级
待正式对外发布
# 配置
1. 鉴权通过OAuth2.0, 所以这里要先申请用户的`appid`、`api_key`、`secret_key`
2. 配置方式一：直接通过制定参数的方式实例化api_client， 参考快速开始部分
3. 配置方式二: yaml配置文件， 配置的默认路径为`~./.config/deepwisdom/dwconfig.yaml`, 配置内容如下
```yaml
api_key: "xx"
secret_key: "xxx"
appid: 3
domain: "xxx"
```

# 快速开始
```python
import deepwisdom as dw

if __name__ == "__main__":
    # 初始化api客户端， 传入申请的appid, api_key, secret_key
    api_client = dw.Client(appid=4, api_key="xx", secret_key="xxx")
    dw.set_client(client=api_client)

    # 从本地数据文件， 上传数据集
    dataset = dw.Dataset.create_from_file(filename="xxx", modal_type=0) #filename为本地数据集文件path

    # 从现有的数据集，创建表格二分类的项目
    primary_label = "is_marry"  # 测试数据源的预测列
    project_name = "SDK-MAIN-PROCESS-TEST"
    train_setting = dw.TrainSetting(training_program="deepwisdom", max_trials=3)  # 高级设置-训练参数设置
    ss = dw.SearchSpace.create(0, 0)  # 获取对应模态的搜索空间
    settings = dw.AdvanceSetting("off", "ga", 6571, target_train=train_setting, search_space=ss.search_space_info)  # 高级设置
    dataset_id = dataset.dataset_id
    project = dw.Project.create_from_dataset(name=project_name, dataset_id=dataset_id, modal_type=0, task_type=0,
                                             scene=1, primary_label=primary_label, primary_main_time_col="",
                                             advance_settings=settings, search_space_id=ss.search_space_id)
    # 训练
    project.wait_train()
    solutions = project.solution_list()
    solution_one = solutions[0]  # 获取推荐的方案

    models = project.get_select_models(solution_one.trial_no, solution_one.trial_type)
    model_one = models[0]  # 获取推荐的模型

    # 上传离线预测数据集
    predict_dataset = project.upload_predict_dataset(filename="xxx")  #filename为本地数据集文件path
    # 离线预测
    offline_predict = dw.OfflinePrediction.predict(model_one.model_id, predict_dataset.dataset_id)
    offline_predict.wait_for_result()
    predict_detail = offline_predict.get_predict_detail(offline_predict.offline_id)  # 离线预测结果

    # 在线推理
    req = dw.CreateDeployRequest(project.project_id, model_one.model_id, "sdk-test", 2, 3, 1, 1)
    deploy = dw.Deployment.create_deployment(req)

    # api详情
    api_info = deploy.get_service_api(deploy.id)
    # 调用api
    resp = deploy.call_service({})

```
# 特性
1. 数据集管理。 包括数据集的增删改查、数据集模糊搜索等
2. 项目管理。 项目的增删改查、训练管理、离线预测、高级设置更新、方案/部署模型列表等
3. 实验管理。 实验详情数据查询，包括耗时、性能和效果指标等
4. 最佳方案。 实验的方案列表及对应的部署模型信息等
5. 离线预测。 获取离线预测列表，进行离线预测等
6. 推理部署。推理服务创建，获取列表，修改常驻状态，修改推理服务名称，调用服务等

# 详细文档
1. API Reference。 
2. tutorials

