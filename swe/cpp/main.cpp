#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <thread>

#define PRINTVAL(comment, x) std::cout << comment << " " << x << "\n";
#define PRINT(comment) std::cout << comment << "\n";
#define LINE_BREAK std::cout << "\n";

void references() {
    int a = 5;
    int& ref = a;
    PRINTVAL("Initially the value of a is", a)
    ref = 2;
    PRINTVAL("After changing a via the reference the value is", a)
    int b = 8;
    PRINTVAL("Initially the value of b is", b)
    ref = b; // really equivalent to a = b
    PRINTVAL("Doing ref = b doesn't change the reference, it sets a to the value of b, i.e.", a)
}

// this is a Player class
class Player {
public:
    int x, y;
    int speed;
    
    void move(int xa, int ya) {
        this->x = this->speed * xa;
        this->y = this->speed * ya;
    }
};
    
void classes() {
    Player player; // this an instance of the Player class, or a new Player object
    player.x = 1;
    player.y = 2;
    player.speed = 5;
    PRINTVAL("Initially, player's x position is", player.x)
    PRINTVAL("Initially, player's y position is", player.y)
    PRINTVAL("Player's speed is", player.speed)
    player.move(2,3);
    PRINT("After calling player.move(2,3)")
    PRINTVAL("Initially, player's x position is", player.x)
    PRINTVAL("Initially, player's y position is", player.y)
}

// this will only compile if s_Variable in static.cpp is decorated with the static keyword
int s_Variable = 5;

void staticKeywordVariable() {
    // this will compile even without static keyword in static.cpp because it is local to staticKeyword function
    // for a global and local variable with the same name, local scope takes priority over global scope
    int s_Variable = 5;
}

struct StaticEntity {
    static int x, y;
    
    // this only works if x and y are decorated with the static keyword
    // otherwise, we get compile error: illegal reference to non-static member 'StaticEntity::x'
    // this is because static methods do not have a reference to class instances, so can only access static vars
    static void Print(std::string name) {
        std::cout << name << ", x: " << x << ", y: " << y << "\n";
    }
};

// seems like these need to be declared in global scope; if declared in local scope we
// get compile error: definition or redeclaration illegal in current scope
int StaticEntity::x;
int StaticEntity::y;
    
void staticKeywordClass() {
    PRINT("Initially, we set StaticEntity e1; e1.x = 1; e1.y = 2; and get")
    StaticEntity e1; e1.x = 1; e1.y = 2; e1.Print("e1");
    PRINT("Then, we set StaticEntity e2; e2.x = 2; e2.y = 4; and get")
    StaticEntity e2; e2.x = 2; e2.y = 4; e2.Print("e2");
    PRINT("So next when we print from e1, since x and y are statically declared in class StaticEntity, we get")
    e1.Print("e1");
    PRINT("Since e1 and e2 really point to same variables, we can write StaticEntity::x = 1")
    PRINT("Note: not writing int StaticEntity::x = 1, because redefining, not defining or redeclaring")
    StaticEntity::x = 1;
    PRINT("Then, printing from e1 and e2 gives same thing")
    e1.Print("e1");
    e2.Print("e2");
    PRINTVAL("And really, it's equivalent to doing StaticEntity::Print() twice but with different arguments to Print() i.e. \"e1\" and \"e2\"", 0)
    StaticEntity::Print("e1");
    StaticEntity::Print("e2");
}

class CtorDtor {
public:
    CtorDtor() {
        std::cout << "Ctor called!\n";
    }
    
    ~CtorDtor() {
        std::cout << "Dtor called!\n";
    }
};

void ctorDtor() {
    CtorDtor instance;
    PRINT("We can also manually call the destructor if we want for whatever reason with instance.~CtorDtor();")
    instance.~CtorDtor();
}

class Character {
public:
    float x, y;
    
    Character() {
        x = 0.0f;
        y = 0.0f;
    }
    
    void Move(float xa, float ya) {
        x += xa;
        y += ya;
    }
    
    // we need the virtual decorator otherwise the override keyword in the 
    // Enemy gives a compile error: method with override specifier 'override' did not override any base class methods
    // not free to use virtual tho: need to maintain a vtable to decide dynamic dispatch of which function per class
    // so space complexity to store vtable and time complexity to iterate through vtable
    virtual void PrintPosition() {
        std::cout << "Character position is x: " << x << ", y: " << y << "\n";
    }
};

class Enemy : public Character {
public:
    int health;
    
    Enemy() {
        health = 10;
    }
    
    void PrintHealth() {
        std::cout << "Enemy health is " << health << "\n";
    }
    
    void PrintPosition() override {
        std::cout << "Enemy position is x: " << x << ", y: " << y << "\n";
    }
};

void inheritance() {
    Enemy enemy;
    enemy.PrintPosition();
    enemy.PrintHealth();
    PRINTVAL("Character has two floats, so size of Character via sizeof(Character) is", sizeof(Character))
    PRINTVAL("Enemy has one int plus whatever in Character, so size of Enemy via sizeof(Enemy) is", sizeof(Enemy))
}

void virtualFunction() {
    Enemy enemy;
    enemy.PrintPosition();
    enemy.PrintHealth();
}

class PureVirtual { // called an interface in Java, like a template whereby subclasses are forced to implement
public:
    virtual void PrintName() = false;
};

class Implemented : public PureVirtual {
public:
    void PrintName() override {
        std::cout << "This is an instance of the Implemented class, a subclass of PureVirtual\n";
    }
};

void interfaces() {
    // can't do this, will get compile error: cannot instantiate abstract class
    //PureVirtual pureVirtual;
    Implemented implemented;
    implemented.PrintName();
}

class ConstEntity {
private:
    int m_x;
    mutable int m_y;

public:
    ConstEntity() {
        m_x = 1;
        m_y = 0;
    }
    
    int GetX() const { 
        m_y++; // this only work if the variable is decorated with the mutable keyword
        return m_x;
    }
    
    int GetY() { 
        return m_y;
    }
};

void printConstEntityXValue(const ConstEntity& entity) {
    std::cout << "ConstEntity x value is " << entity.GetX() << "\n";
}

void printConstEntityYValue(ConstEntity& entity) {
    std::cout << "ConstEntity y value is " << entity.GetY() << "\n";
}

void constKeyword() {
    const int MAX_AGE = 90;
    const int* ptrToConstInt = &MAX_AGE;
    
    // error, pointer to const int, so can't dereference and modify the value pointed to
    // error C3892: 'ptrToConstInt': you cannot assign to a variable that is const
    //*ptrToConstInt = 10;
    
    // error: cannot point an int pointer to a const int, need a const int pointer
    // error C2440: 'initializing': cannot convert from 'const int *' to 'int *'
    //int* PtrToInt = &MAX_AGE;
    
    int a = 1;
    int b = 2;
    int* const constPtrToInt = &a;
    
    // error: const pointer cannot be reassigned
    // error C3892: 'constPtrToInt': you cannot assign to a variable that is const
    //constPtrToInt = &b;
    
    // error: const pointer cannot be reassigned
    // error C3892: 'constPtrToConstInt': you cannot assign to a variable that is const
    const int* const constPtrToConstInt = &MAX_AGE;
    //constPtrToConstInt = &b;
    
    ConstEntity entity;
    
    // this function will give a compile error if any of the ConstEntity methods used in the function have not been marked as const
    // e.g. it will not work if we call ConstEntity::GetY() in the function, because this function is not const
    printConstEntityXValue(entity);
    printConstEntityYValue(entity);
}

void mutableLambda() {
    PRINT("Before, x was 420 but after modifying in the lambda function with keyword mutable")
    int x = 420;
    auto f = [&]() mutable {
        x++;
        std::cout << "Printed from lambda: " << x << "\n";
     };
    f();
}

class Example {
private:
    int m_data;

public:
    Example() : m_data(-1) {
        std::cout << "Example constructor called, m_data: " << m_data << "\n";
    }
    
    Example(int data) : m_data(data) {
        std::cout << "Example constructor called, m_data: " << m_data << "\n";
    }
};

class InitializerListClass {
private:
    Example m_example;

public:
    InitializerListClass() {
        m_example = Example(0);
    }
    
    // by using initializer lists, we avoid needlessly calling default constructors
    InitializerListClass(int data) : m_example(data) {}
};

void initializerLists() {
    InitializerListClass noList;
    InitializerListClass withList(1);
}

class ImplicitExplicit {
private:
    std::string m_name;
    int m_age;

public:
    explicit ImplicitExplicit(const std::string name) : m_name(name), m_age(0) {
        std::cout << "ImplicitExplicit name constructor called with " << name << "!\n";
    }
    
    ImplicitExplicit(int age) : m_name("Unknown"), m_age(age) {
        std::cout << "ImplicitExplicit age constructor called with " << age << "!\n";
    }
};

void convertImplicitExplicit(const ImplicitExplicit& impExp) {}

void implicitExplicit() {
    // won't work because implicit conversion isn't allowed for the name constructor
    //ImplicitExplicit impExpName = "Alovya";
    
    // need to explicitly construct
    ImplicitExplicit impExpName("Alovya");
    
    // will work because implicit conversion is allowed for the age constructor
    ImplicitExplicit impExpAge = 27;
    
    convertImplicitExplicit(27); // implicit conversion
    
    // won't work because only at most one implicit conversion is allowed but "Alovya" is a char*, so we'd need
    // firstConversion = std::string("Alovya") to go from char* to std::string, and then ImplicitExplicit(firstConversion)
    //convertImplicitExplicit("Alovya");
}

struct Vec2 {
    float x, y;
    
    Vec2(float x, float y) : x(x), y(y) {}
    
    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }
    
    Vec2 operator*(const Vec2& other) const {
        return Vec2(x * other.x, y * other.y);
    }
};

Vec2 operator*(float lhs, Vec2 rhs) {
    return Vec2(lhs * rhs.x, lhs * rhs.y);
}

Vec2 operator*(Vec2 lhs, float rhs) {
    return rhs * lhs; // reusing left hand multply
}

void operatorOverloading() {
    Vec2 pos(4.0f, 4.0f);
    Vec2 jump(0.5f, 1.5f);
    Vec2 powerup(1.1f, 1.1f);
    Vec2 newPos = pos + jump * powerup;
    Vec2 doublePos = pos * 2;
}

void smartPointers() {
    std::unique_ptr<Example> examplePtr = std::make_unique<Example>(420);
    
    //std::shared_ptr<Example> sharedExamplePtr;
    std::weak_ptr<Example> weakExamplePtr;
    {
        std::shared_ptr<Example> otherSharedExamplePtr = std::make_shared<Example>(20);
        //sharedExamplePtr = otherSharedExamplePtr;
        weakExamplePtr = otherSharedExamplePtr;
    }
}

class String {
public:
    String(const char* string) {
        std::cout << "String constructor called!\n";
        m_size = strlen(string);
        m_buffer = new char[m_size + 1];
        memcpy(m_buffer, string, m_size);
        m_buffer[m_size] = '\0';
    }
    
    String(const String& other) : m_size(other.m_size) {
        std::cout << "String copy constructor called!\n";
        m_buffer = new char[m_size + 1];
        memcpy(m_buffer, other.m_buffer, m_size + 1);
    }
    
    String(String&& other) : m_size(other.m_size) {
        std::cout << "String move constructor called!\n";
        m_buffer = other.m_buffer;
        other.m_size = 0;
        other.m_buffer = nullptr;
    }
    
    String& operator=(String&& other) {
        if (this != &other) {
            std::cout << "String move assignment operator called!\n";
            delete[] m_buffer;
            m_size = other.m_size;
            m_buffer = other.m_buffer;
            other.m_size = 0;
            other.m_buffer = nullptr;
        }
        
        return *this;
    }
    
    void Print() {
        std::cout << m_buffer << "\n";
    }
    
    ~String() {
        std::cout << "String destructor called!\n";
        delete[] m_buffer;
    }
    
    friend std::ostream& operator<<(std::ostream& stream, const String& string);
    
    char& operator[](int idx) {
        return m_buffer[idx];
    }

private:
    char* m_buffer;
    int m_size;
};

std::ostream& operator<<(std::ostream& stream, const String& string) {
    stream << string.m_buffer;
    return stream;
}

class Entity {
public:
    // calls copy constructor in m_name(name) if no move constructor available for class String
    Entity(const String& name) : m_name(name) {}
    
    Entity(String&& name) : m_name(std::move(name)) {}
    
    void PrintName() {
        std::cout << m_name << "\n";
    }

private:
    String m_name;
};

void moveSemantics() {
    Entity entity(String("Alovya"));
    entity.PrintName();
}

void moveAssignment() {
    String src1("src1");
    String src2("src2");
    // casting with (String&&)
    // also, this is implicit conversion, i.e. dst1((String&&)src) is implicitly called
    String dst1 = (String&&)src1;
    // instead of casting with (String&&), use std::move
    // also, this is creating a new object, not assigning to an existing one
    // i.e. move constructor is called, but not move assignment operator
    String dst2 = std::move(src2);
    
    String src3("src3");
    PRINT("Initially, the name of String src3(\"src3\") is")
    src3.Print();
    
    // this is assignment to an existing variable
    dst2 = std::move(src3);
    
    PRINT("Then we do dst2 = std::move(src3) whose name is")
    dst2.Print();
    PRINT("So the name of String src(\"src\") has been taken by dst2 and is now")
    src3.Print();
}

void printByReference(const String& string) {
    std::cout << string << "\n";
}

void printByValue(const String string) {
    std::cout << string << "\n";
}

void copyCtor() {
    String string("Alovya");
    std::cout << string << "\n";
    String copy = string;
    copy[2] = 'a';
    std::cout << copy << "\n";
    printByReference(copy);
    printByValue(copy);
}

void localStaticIncrementor() {
    static int a = 0;
    a++;
    std::cout << a << "\n";
}

void localStatic() {
    localStaticIncrementor();
    localStaticIncrementor();
    localStaticIncrementor();
    localStaticIncrementor();
    localStaticIncrementor();
}

void printHelloWorld(std::string message) {
    std::cout << "Hello world! This is my message: " << message << "\n";
}

typedef void(*FuncPtr)(std::string);

void printValue(int value) {
    std::cout << value << "\n";
}

typedef void(*IntFuncPtr)(int);

void forEach(const std::vector<int>& values, IntFuncPtr func) {
    for (int value : values) {
        func(value);
    }
}

void functionPointers() {
    printHelloWorld("raw function");
    auto autoFuncPtr = &printHelloWorld;
    autoFuncPtr("auto function pointer");
    FuncPtr funcPtr = &printHelloWorld;
    funcPtr("typedef'd");
    std::vector<int> values = {1,2,3,4,5,6};
    forEach(values, printValue);
    forEach(values, [](int value) { std::cout << value << "\n"; });
}

void lambdaFunctions() {
    int offset = 420;
    std::function<int(int)> lambdaPtr = [offset](int value) -> int { return offset + value; };
    PRINTVAL("The value return by calling lambdaPtr() is", lambdaPtr(69000))
}

static bool s_Finished = false;

void threads() {
    std::thread worker([]() -> void { while (!s_Finished) { std::cout << "Working...\n"; }; });
    std::cin.get();
    s_Finished = true;
    worker.join(); // waiting until this thread joins, i.e. finishes executing
    std::cin.get();
}

class Base {
public:
    Base() { std::cout << "Base constructed!\n"; }
    virtual ~Base() { std::cout << "Base destructed!\n"; }
};

class Derived : public Base {
public:
    Derived() { std::cout << "Derived constructed!\n"; }
    ~Derived() { std::cout << "Derived destructed!\n"; }
};

void virtualDestructor() {
    Base* base = new Base();
    delete base;
    std::cout << "------------\n";
    Derived* derived = new Derived();
    delete derived;
    std::cout << "------------\n";
    Base* poly = new Derived();
    delete poly;
}

void casting() {
    // static_cast<>() e.g. double = static_cast<double>(int)
    // const_cast<>() e.g. change constness of variable
    // dynamic_cast<>() e.g. for polymorphism
    // reinterpret_cast<>() e.g. for literal reading of bytes a la type punning e.g. reinterpret_cast<float4*>(float*)
}

int& returnValue(int& a) {
    return a;
}

void takeValue(int& val) {}

void takeValue(int&& val) {}

void printString(std::string&& message) {
    std::cout << "[rvalue] " << message << "\n";
}

void printString(std::string& message) {
    std::cout << "[lvalue] " << message << "\n";
}

void lvaluesAndrvalues() {
    int i = 10;
    // this won't work if an rvalue is returned by returnValue
    returnValue(i) = 10;
    // this won't work because int& 10 doesn't make sense
    // need rvalue reference overload, int&&, or const int&
    takeValue(10);
    
    std::string hello = "hello";
    std::string world = " world!";
    std::string helloWorld = hello + world;
    
    printString(helloWorld);
    
    // (hello + world) -> is actually an lvalue temporary
    // so won't work without std::string&& (l-value reference)
    printString(hello + world);
}



int main(int argc, char** argv) {
    references();
    LINE_BREAK
    classes();
    LINE_BREAK
    staticKeywordVariable();
    LINE_BREAK
    staticKeywordClass();
    LINE_BREAK
    ctorDtor();
    LINE_BREAK
    inheritance();
    LINE_BREAK
    interfaces();
    LINE_BREAK
    constKeyword();
    LINE_BREAK
    mutableLambda();
    LINE_BREAK
    initializerLists();
    LINE_BREAK
    implicitExplicit();
    LINE_BREAK
    operatorOverloading();
    LINE_BREAK
    smartPointers();
    LINE_BREAK
    copyCtor();
    LINE_BREAK
    localStatic();
    LINE_BREAK
    functionPointers();
    LINE_BREAK
    lambdaFunctions();
    LINE_BREAK
    //threads();
    //LINE_BREAK
    virtualDestructor();
    LINE_BREAK
    casting();
    LINE_BREAK
    lvaluesAndrvalues();
    LINE_BREAK
    moveSemantics();
    LINE_BREAK
    moveAssignment();
    LINE_BREAK
}

/*
std::move
*/