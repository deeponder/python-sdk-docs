# 主流程
```python
import time
import unittest
import deepwisdom as dw

dataset_file_path = "/Users/up/Downloads/"
dataset_file_name = "data_upload_test.csv"
train_succ_status = 2

class TestMainProcess(unittest.TestCase):
    def test_main_process(self):
        api_client = dw.Client(appid=4, api_key="RrTLKoGrgKRXkSJAstcndNLa",
                               secret_key="xJHb3TjOxh1cqVb0seLBEpHDWLA3fYE7", domain="http://192.168.50.122:30772")
        dw.set_client(client=api_client)
        # 数据集
        dataset = dw.Dataset.create_from_file(dataset_file_path+dataset_file_name, 0)
        self.assertEqual(dataset.dataset_name, dataset_file_name)
        # 项目
        primary_label = "is_marry"
        project_name = "SDK-MAIN-PROCESS-TEST"
        train_setting = dw.TrainSetting(training_program="zhipeng", max_trials=3)
        ss = dw.SearchSpace.create(0, 0)
        ss.custom_model_hp(["LIGHTGBM", "CATBOOST"])
        settings = dw.AdvanceSetting("off", "ga", 6571, target_train=train_setting, search_space=ss.search_space_info)
        dataset_id = dataset.dataset_id
        # dataset_id = 6062
        project = dw.Project.create_from_dataset(name=project_name, dataset_id=dataset_id, model_type=0, task_type=0,
                                                 scene=1, primary_label=primary_label, primary_main_time_col="", id_cols="",
                                                 advance_settings=settings, search_space_id=ss.search_space_id)
        self.assertEqual(project.name, project_name)
        ## 训练
        project.wait_train()
        self.assertEqual(train_succ_status, project.status)
        solutions = project.solution_list()
        solution_one = solutions[0]
        self.assertEqual(solution_one.project_id, project.project_id)

        models = project.get_select_models(solution_one.trial_no, solution_one.trial_type)
        model_one = models[0]
        self.assertEqual(model_one.project_id, project.project_id)

if __name__ == '__main__':
    unittest.main()


```