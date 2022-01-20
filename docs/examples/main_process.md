# 主流程
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