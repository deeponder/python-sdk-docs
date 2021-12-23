目前可简单参考如下主流程的用例

```python
import time
import unittest
import deepwisdom as dw

dataset_file_path = "/Users/up/Downloads/"
dataset_file_name = "data_upload_test.csv"
train_succ_status = 2


class TestMainProcess(unittest.TestCase):
    def test_main_process(self):
        # 初始化api_client
        api_client = dw.Client(appid=4, api_key="RrTLKoGrgKRXkSJAstcndNLa",
                               secret_key="xJHb3TjOxh1cqVb0seLBEpHDWLA3fYE7", domain="http://192.168.50.122:30772")
        dw.set_client(client=api_client)
        # 数据集上传
        dataset = dw.Dataset.create_from_file(dataset_file_path+dataset_file_name, 0)
        self.assertEqual(dataset.name, dataset_file_name)
        time.sleep(60)  #这里需要等dataset都处理完（包括eda啥的）  @chucheng
        # 项目创建
        primary_label = "is_marry"  # 预测列
        project_name = "SDK-MAIN-PROCESS-TEST"  # 项目名称
        train_setting = dw.TrainSetting(training_program="zhipeng", max_trials=3)   # 一些高级设置
        settings = dw.AdvanceSetting("off", "ga", 6571, target_train=train_setting)
        dataset_id = dataset.dataset_id
        project = dw.Project.create_from_dataset(name=project_name, dataset_id=dataset_id, model_type=0, task_type=0,
                                         scene=1, primary_label=primary_label, primary_main_time_col="", id_cols="", advance_settings=settings)
        self.assertEqual(project.name, project_name)
        ## 开始模型训练
        project.wait_train()
        self.assertEqual(train_succ_status, project.status)
        ## 方案列表
        solutions = project.solution_list()
        solution_one = solutions[0]
        self.assertEqual(solution_one.project_id, project.project_id)
        ## 模型列表
        models = project.get_select_models(solution_one.trial_no, solution_one.trial_type)
        model_one = models[0]
        self.assertEqual(model_one.project_id, project.project_id)
        
        ## 服务部署
        deployment = dw.Deployment.create_deployment({
            "project_id": project.project_id,
            "model_inst_id": model_one.model_id,
            "name": "灵魂拷问--晚上吃什么",
            "gpu_num": 1,
            "gpu_mem": 2,
            "memory_limit": 2,
            "min_pod": 1,
            "max_pod": 2,
        })
        rsp_body = deployment.call_service({})
        self.assertIsNotNone(rsp_body)
        
        ## 离线预测
        pred = dw.OfflinePrediction.predict_by_model_dataset(model_one.model_id,dataset.dataset_id)
        self.assertIsNotNone(pred)
        
        

if __name__ == '__main__':
    unittest.main()


```