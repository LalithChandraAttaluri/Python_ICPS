class Flight:
    def __init__(self, airline_name, flight_number):                                        #Default constructor for flight class
        self.airline_name = airline_name
        self.flight_number = flight_number

    def flight_display(self):                                                               #Displaying flight details
        print('Airlines : ', self.airline_name)
        print('Flight number : ', self.flight_number)


class employee:                                                                             #eloyee class
    def __init__(self, e_id, e_name, e_age, e_gender):                              #eloyee class constructor
        self.e_name = e_name
        self.e_age = e_age
        self.__e_id = e_id
        self.e_gender =  e_gender
    def e_display(self):                                                            #Displaying eloyee details
        print("Name of employee: ",self.e_name)
        print('employee id: ', self.__e_id)
        print('employee age: ',self.e_age)
        print('employee gender: ', self.e_gender)

class Passenger:                                                                            #Passenger class
    def __init__(self):
        Passenger.__passport_number = input("Enter the passport number of the passenger: ") #Passport number is declared as private data member
        Passenger.name = input('Enter name of the passenger: ')
        Passenger.age = input('Enter age of passenger : ')
        print("Please select the gender from the below list")
        print(""" 
                  1. Male
                  2. Female
                  3. Transgender
                  4. Don't like to mention
                  """)
        choice = int(input("Enter Choice:"))
        if choice == 1:
            Passenger.gender = "Male"
        elif choice == 2:
            Passenger.gender = "Female"
        elif choice == 3:
            Passenger.gender = "Transgender"
        elif choice == 4:
            Passenger.gender = "Passenger dont want to reveal the gender"


class Baggage():                                                                            #Baggage class
    checked_bags = 3
    bag_fare = 0
    def __init__(self, checked_bags):
        self.checked_bags = checked_bags
        if self.checked_bags > 2 :                                                               #Calculating the cost if there are more than two cabin bags
            for i in range(self.checked_bags):
                self.bag_fare += 100
        print("Number of checked bags allowed: ",checked_bags,"bag fare: ",self.bag_fare)


class Fare(Baggage):                                                                        #Fare class which is subclass of Baggage
    counter = 150                                                                           #Cost is fixed for purchasing at counter
    online = 300                                                       #Cost varies with ticket is purchased through online and fair is gdouble the counter price
    total_fare=0
    def __init__(self):
        checkin_bags=input('Please enter the number of Check-in bags you like to carry: ')
        checkin_bags = int(checkin_bags)
        super().__init__(checkin_bags)                                                                 #Super call
        print("Please select the type of booking")
        print(""" 
                                  1. Online
                                  2. Counter
                                  """)
        choice2 = int(input("Enter Choice:"))
        if choice2 == 1:
            Fare.total_fare = self.online + self.bag_fare
        elif choice2 == 2:
            Fare.total_fare = self.counter + self.bag_fare
        print("Total Fare before class type: ",Fare.total_fare)


class Ticket(Passenger, Fare):                                                             #Multiple inheritence
    def __init__(self):
        print("Please select the class you like to travell from below")
        print(""" 
                                          1. Bussiness
                                          2. Economy
                                          """)
        choice1 = int(input("Enter Choice:"))
        if choice1 == 1:
            Passenger.class_type = "Bussiness"
            Final_ticket_price=Fare.total_fare+ 100
        else:
            Passenger.class_type = "Economy"
            pass
        print("Passenger name:",Passenger.name)                                            #Acccessing parent class variable
        print("Passenger age:", Passenger.age)
        print("Passenger gender:", Passenger.gender)
        print("Passenger class:", Passenger.class_type)
        print("Final ticket fare is:",Final_ticket_price)                                              #Displaying total fair for itenary


f1=Flight('American Airlines',6789)
f1.flight_display()

e0 = employee('e1', 'e_Anu', 21, 'F')
e0.e_display()

p1 = Passenger()

fare1=Fare()

t= Ticket()