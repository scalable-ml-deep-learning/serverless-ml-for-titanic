import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("titanic_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("scalable"))
   def f():
       g()


def generate_passenger(survived):
    """
    Returns a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random
    if survived :
        df = pd.DataFrame({ "pclass": [random.choice([2,1])],
                        "sex": [random.choice([0,1])],
                        "age": [random.uniform(0, 99)],
                        "fare": [random.uniform(50, 500)]
                        })
    else:
        df = pd.DataFrame({ "pclass": [random.choice([3,2])],
                        "sex": [random.choice([0,1])],
                        "age": [random.uniform(0, 99)],
                        "fare": [random.uniform(0, 50)]
                        })
    df['survived'] = survived
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    
    survived_df = generate_passenger(1)
    dead_df = generate_passenger(0)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        passenger_df = survived_df
        print("Survived added")
    else:
        passenger_df = dead_df
        print("Dead added")

    return passenger_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    passenger_df = get_random_passenger()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=1)
    titanic_fg.insert(passenger_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic_daily")
        with stub.run():
            f()
