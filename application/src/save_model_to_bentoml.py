import bentoml
import hydra
import joblib
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
import xgboost as xgb


def load_model(model_path: str):
    booster= xgb.Booster()
    return booster.load_model(model_path)
    # return joblib.load(model_path)



@hydra.main(config_path="../../config", config_name="main")
def save_to_bentoml(config: DictConfig):
    print("Model_path: ",config.model.path)
    print("Model_name: ",config.model.name)
    model = load_model(abspath(config.model.path))
    print("MODEL: ",model)
    # bentoml.picklable_model.save_model(config.model.name, model)
    bentoml.xgboost.save_model(config.model.name, model)


if __name__ == "__main__":
    save_to_bentoml()
