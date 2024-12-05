import streamlit as st

def render_add_widget(_conn, _cursor):
    """Render the form for adding a new record."""

    general = {
        "is_electric": False,
        "year": 2023,
        "make": "",
        "model": "",
        "price": 0.0,
        "mileage": 0.0,
        "transmission": "",
        "owners": 0,
        "VIN": "",
        "accidents": "",
        "open_recalls": 0,
        "cylinders": 0,
        "drive_type": "",
        "bed_length": 0.0,
        "exterior_color": "",
        "interior_color": "",
        "class": "",
        "is_auction": False,
        "has_odometer_issue": False,
        "certification": ""
    }

    gas_car = {
        "fuel": "",
        "miles_per_gallon": 0.0,
    }

    electric_car = {
        "miles_per_gallon_equivalent": 0.0,
        "range_when_new": 0.0,
        "time_to_fully_charge": 0.0
    }

    # Render the form
    with st.form("add_new_car_form"):
        st.subheader("Add New Car")

        # general info for both gas and electric car
        st.write("**General Information**")
        general["is_electric"] = st.checkbox("Is Electric?", value=False)
        general["year"] = st.number_input("Year", value=general["year"], min_value=1886, max_value=2100)
        general["make"] = st.text_input("Make", value=general["make"])
        general["model"] = st.text_input("Model", value=general["model"])
        general["price"] = st.number_input("Price", value=general["price"], min_value=0.0)
        general["mileage"] = st.number_input("Mileage", value=general["mileage"], min_value=0.0)
        general["transmission"] = st.text_input("Transmission", value=general["transmission"])
        general["owners"] = st.number_input("Owners", value=general["owners"], min_value=0)
        general["VIN"] = st.text_input("VIN", value=general["VIN"])
        general["accidents"] = st.text_input("Accidents Condition", value=general["accidents"])
        general["open_recalls"] = st.number_input("Number of Open Recalls", value=general["open_recalls"], min_value=0)
        general["cylinders"] = st.number_input("Cylinders", value=general["cylinders"], min_value=0)
        general["drive_type"] = st.text_input("Drive Type", value=general["drive_type"])
        general["bed_length"] = st.number_input("Bed Length", value=general["bed_length"], min_value=0.0)
        general["exterior_color"] = st.text_input("Exterior Color", value=general["exterior_color"])
        general["interior_color"] = st.text_input("Interior Color", value=general["interior_color"])
        general["class"] = st.text_input("Class", value=general["class"])
        general["is_auction"] = st.checkbox("Is Auctioned?", value=general["is_auction"])
        general["has_odometer_issue"] = st.checkbox("Has Odometer Issue?", value=general["has_odometer_issue"])
        general["certification"] = st.text_input("Certification", value=general["certification"])

        # if is_electric==True, then feature entered for gas car won't be uploaded and vice versa
        if general["is_electric"]:
            st.write("**Electric Car Information**")
            electric_car["miles_per_gallon_equivalent"] = st.number_input("Miles Per Gallon Equivalent", value=electric_car["miles_per_gallon_equivalent"], min_value=0.0)
            electric_car["range_when_new"] = st.number_input("Range When New", value=electric_car["range_when_new"], min_value=0.0)
            electric_car["time_to_fully_charge"] = st.number_input("Time to Fully Charge (hours)", value=electric_car["time_to_fully_charge"], min_value=0.0)
        else:
            st.write("**Gas Car Information**")
            gas_car["fuel"] = st.text_input("Fuel Type", value=gas_car["fuel"])
            gas_car["miles_per_gallon"] = st.number_input("Miles Per Gallon", value=gas_car["miles_per_gallon"], min_value=0.0)

      
        submitted = st.form_submit_button("Add New Car")
        if submitted:
            add_new_record(_conn, _cursor, general, gas_car if not general["is_electric"] else None, electric_car if general["is_electric"] else None)
            st.session_state["record_added"] = True
            
def add_new_record(_conn, _cursor, general, gas, electric):
    query = """
        INSERT INTO used_cars (
            is_electric, year, make, model, price, mileage, transmission, owners,
            VIN, accidents, open_recalls, cylinders, drive_type, bed_length,
            exterior_color, interior_color, class, is_auction, has_odometer_issue,
            certification, fuel, miles_per_gallon, miles_per_gallon_equivalent,
            range_when_new, time_to_fully_charge
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = [
        general["is_electric"], general["year"], general["make"], general["model"], general["price"],
        general["mileage"], general["transmission"], general["owners"], general["VIN"],
        general["accidents"], general["open_recalls"], general["cylinders"], general["drive_type"],
        general["bed_length"], general["exterior_color"], general["interior_color"],
        general["class"], general["is_auction"], general["has_odometer_issue"],
        general["certification"]
    ]

    # Add gas or electric-specific fields
    if gas:
        values.extend([gas["fuel"], gas["miles_per_gallon"], None, None, None])
    elif electric:
        values.extend([None, None, electric["miles_per_gallon_equivalent"], electric["range_when_new"], electric["time_to_fully_charge"]])

    # Execute the query
    try:
        _cursor.execute(query, values)
        _conn.commit()
        st.success("The new record has been added successfully!")
    except Exception as e:
        st.error(f"Error Found: {e}")