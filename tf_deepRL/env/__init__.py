from tf_deepRL.env.core import Env
from tf_deepRL.env.openai_gym import gymEnv

env_providers = ['openai_gym']

def load_env(provider="openai_gym", name=None):
    if provider in env_providers:
        pass
    else:
        err_txt = ""
        for prov in env_providers: err_txt += prov + ", "
        raise KeyError("This environment provider is not supported. " + "Supported environment providers: " + err_txt[:-2])

    if provider == "gym"
        env = gymEnv(name)

    return env
