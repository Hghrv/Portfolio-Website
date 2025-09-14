import psycopg2
from psycopg2 import sql

def update_database(train_set_x, train_set_y, test_set_x, test_set_y):
    try:
        # Connecting to the default 'postgres' database
        connection = psycopg2.connect(
            dbname="postgres",
            user="hgrv",
            password="hgrv_pass",
            host="localhost",
            port="5432"
        )
        connection.autocommit = True  # Enable autocommit for database creation

        # Creating a cursor object for SQL script
        cursor = connection.cursor()

        # Defining a name for the database for the first instance            
        database_name = "data_warehouse"
                
        # Checking whether the database already exists
        cursor.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [database_name])
        exists = cursor.fetchone()

        # Creating the database for the first instance
        if not exists: 
            # Executing a SQL command to create the database if it does not already exist (first instance only)
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
            print(f"Database '{database_name}' created successfully!")
        else:
            print(f"Database '{database_name}' already exists.")
                
        # SQL query to create the tables
        create_table_query = """
            CREATE TABLE IF NOT EXISTS training_data (
                id SERIAL PRIMARY KEY,                  
                x_coordinates NUMERIC(10, 2),
                y_coordinates NUMERIC(10, 2),
                timestamp DATE
            );
            CREATE TABLE IF NOT EXISTS test_data (
                id SERIAL PRIMARY KEY,                  
                x_coordinates NUMERIC(10, 2),
                y_coordinates NUMERIC(10, 2),
                timestamp DATE DEFAULT CURRENT_DATE
            );                
            """

        # Executing the query              
        cursor.execute(create_table_query)
        connection.commit()  # Save changes to the database
        print("Table 'employees' created successfully!")
            
        # Inserting x-coordinates, and timestamp into table 'training_data'
        for item in train_set_x:
            # Data to insert
            training_data_to_insert = item

            # SQL query to insert data into tables
            insert_query = """
            INSERT INTO training_data (x_coordinates, timestamp)
            VALUES (train_set_x, timestamp_current)
            """
            
            # Execute the query
            cursor.execute(insert_query, training_data_to_insert)

            # Commit the SQL script
            connection.commit()              
                    
        # Inserting y-coordinates into table 'training_data'
        for item in train_set_y:
            # Data to insert
            training_data_to_insert = item

            # SQL query to insert data into tables
            insert_query = """
            INSERT INTO training_data (y_coordinates)
            VALUES (item, timestamp_current)
            """
            
            # Execute the query
            cursor.execute(insert_query, training_data_to_insert)

            # Commit the SQL script
            connection.commit()              
                    
        # Inserting x-coordinates, and timestamp into table 'test_data'
        for item in test_set_x:
            # Data to insert
            test_data_to_insert = item

            # SQL query to insert data into tables
            insert_query = """
            INSERT INTO test_data (x_coordinates, timestamp)
            VALUES (item, timestamp_current)
            """
            
            # Execute the query
            cursor.execute(insert_query, test_data_to_insert)

            # Commit the SQL script
            connection.commit()              
                    
        # Inserting y-coordinates into table 'test_data'
        for item in train_set_y:
            # Data to insert
            test_data_to_insert = item

            # SQL query to insert data into tables
            insert_query = """
            INSERT INTO test_data (y_coordinates)
            VALUES (item, timestamp_current)
            """
            
            # Execute the query
            cursor.execute(insert_query, test_data_to_insert)

            # Commit the SQL script
            connection.commit()  

        print("Data inserted successfully into {data_warehouse}!")
            
    except Exception as e:
        print(f"Error: {e}")
        print(f"Check for any error in the SQL script.")
    finally:
        # Closing the cursor and connection
        cursor.close()
        connection.close()
    print("PostgreSQL connection is now closed.")       
