
# 数据(Dataset)

## 从数据源创建数据集
```python
datasets = dw.Dataset.create_from_data_source(
            '{"host":"192.168.50.36","port":"10000","user":"root","password":"","db":"default"}',  #mysql/hive
            0,  #云类型: 0本地, 1Amazon, 2阿里云, 3腾讯云, 4华为云
            5,  #数据来源: 0本地文件, 1mysql, 2oracle, 3mariadb, 4hdfs, 5hive
            '[{"autotables":[{"table_name":"dataset_update_record"},{"table_name":"scene"}]}]'  # 选择的table 列表。 autotables代表db，dataset_update_record|scene代表autotable下的两个表,支持多层嵌套 
        )
```
