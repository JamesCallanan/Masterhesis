import psycopg2

def insert_tuner_search(search, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute(f"""INSERT INTO tuner_search 
                        ( search_id ,
                          search_type,
                          num_models,
                          num_epochs,
                          model_template_builder_name,
                          hyperparam_ranges,
                          disease_classes,
                          model_mode,
                          perform_ROI,
                          depth,
                          width,
                          height,
                          git_commit_id,
                          git_branch,
                          tensorboard_folder_path,
                          keras_tuner_folder_path 
                        ) 
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT DO NOTHING
                        """, 
                        ( 
                          search['search_id'],
                          search['search_type'],
                          search['num_models'],
                          search['num_epochs'],
                          search['model_template_builder_name'],
                          search['hyperparam_ranges'],
                          search['disease_classes'],
                          search['model_mode'],
                          search['perform_ROI'],
                          search['depth'],
                          search['width'],
                          search['height'],
                          search['git_commit_id'],
                          search['git_branch'],
                          search['tensorboard_folder_path'],
                          search['keras_tuner_folder_path'] 
                        )                       
                       )
    conn.close()


def insert_trial(trial, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute(f"""INSERT INTO trials 
                          ( trial_id,
                            search_id,
                            model_path,
                            val_loss,
                            val_acc,
                            train_loss,
                            train_acc,
                            last_conv_layer_name
                          )
                          VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                          ON CONFLICT DO NOTHING
                        """,
                        (
                          trial['trial_id'],
                          trial['search_id'],
                          trial['model_path'],
                          trial['val_loss'],
                          trial['val_acc'],
                          trial['train_loss'],
                          trial['train_acc'],
                          trial['last_conv_layer_name']
                        )
                      )
    conn.close()


def update_trial_with_heatmap_data(trial_updated, database_connection_details): 
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute( """ UPDATE trials 
                            SET 
                              average_fraction_of_heart_in_mri_batch = %s ,
                              average_fraction_of_pos_gradients_in_heart_in_batch_of_mris = %s ,
                              average_fraction_of_neg_gradients_in_heart_in_batch_of_mris = %s
                            WHERE 
                            trial_id = '%s' 
                        """,
                        ( 
                            trial_updated['average_fraction_of_heart_in_mri_batch'],
                            trial_updated['average_fraction_of_pos_gradients_in_heart_in_batch_of_mris'],
                            trial_updated['average_fraction_of_neg_gradients_in_heart_in_batch_of_mris'],
                            trial_updated['trial_id']
                        )
                      )
    conn.close()


def get_trial_by_id(trial_id, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
      cursor.execute("SELECT * FROM trials WHERE trial_id = '%s'", (trial_id,))
      results = cursor.fetchall()
    conn.close()
    return results


def get_tuner_search_by_id(search_id, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
      cursor.execute(f"SELECT * FROM tuner_search WHERE search_id = '%s'", (search_id,))
      results = cursor.fetchall()
    conn.close()
    return results

def get_all_trials(database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute("SELECT * FROM trials")
        results = cursor.fetchall()
    conn.close()
    return results

def get_all_tuner_searches(database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute("SELECT * FROM tuner_search")
        results = cursor.fetchall()
    conn.close()
    return results