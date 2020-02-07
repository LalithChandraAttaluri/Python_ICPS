class Employee:
    emp_count=0
    emp_total_salary=0
    def __init__(self,name,family,salary,department):
        self.emp_name=name
        self.emp_family=family
        self.emp_department=department
        self.emp_salary=salary
        Employee.emp_count+=1
        Employee.emp_total_salary = Employee.emp_total_salary+self.emp_salary

    def display_employee_details(self):
        print('employee name:',self.emp_name)
        print('employee Family:', self.emp_family)
        print('Employee Department:', self.emp_department)
        print('Employee Salary:', self.emp_salary)


class FullTimeEmployee(Employee):
    def __init__(self, name, family, salary, department):
        Employee.__init__(self, name, family, salary, department)
    def display_employee_details(self):
        print('employee name:',self.emp_name)
        print('employee Family:', self.emp_family)
        print('Employee Department:', self.emp_department)
        print('Employee Salary:', self.emp_salary)

f1 = FullTimeEmployee("Pranu", "Mutha", 600000, "CS")
f2 = FullTimeEmployee("Sirisha", "Rella", 500000, "CS")
f3 = FullTimeEmployee("Nikki", "Pateel", 200000, "CS")

f1.display_employee_details()
print('-------------------------')
f2.display_employee_details()
print('-------------------------')
f3.display_employee_details()
print('-------------------------')

print('The total number of employees are:',FullTimeEmployee.emp_count)
print('-------------------------')
print('The average salary of the given employees is:', FullTimeEmployee.emp_total_salary/FullTimeEmployee.emp_count)
