#include <iostream>

#define PRINT(comment, x) std::cout << comment << " " << x << "\n";

#define LINE_BREAK std::cout << "\n";

void references() {
    int a = 5;
    int& ref = a;
    PRINT("Initially the value of a is", a)
    ref = 2;
    PRINT("After changing a via the reference the value is", a)
    int b = 8;
    PRINT("Initially the value of b is", b)
    ref = b; // really equivalent to a = b
    PRINT("Doing ref = b doesn't change the reference, it sets a to the value of b, i.e.", a)
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
    PRINT("Initially, player's x position is", player.x)
    PRINT("Initially, player's y position is", player.y)
    PRINT("Player's speed is", player.speed)
    player.move(2,3);
    PRINT("After calling player.move(2,3)", 0)
    PRINT("Initially, player's x position is", player.x)
    PRINT("Initially, player's y position is", player.y)
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
    PRINT("Initially, we set StaticEntity e1; e1.x = 1; e1.y = 2; and get", 0)
    StaticEntity e1; e1.x = 1; e1.y = 2; e1.Print("e1");
    PRINT("Then, we set StaticEntity e2; e2.x = 2; e2.y = 4; and get", 0)
    StaticEntity e2; e2.x = 2; e2.y = 4; e2.Print("e2");
    PRINT("So next when we print from e1, since x and y are statically declared in class StaticEntity, we get", 0)
    e1.Print("e1");
    PRINT("Since e1 and e2 really point to same variables, we can write StaticEntity::x = 1", 0)
    PRINT("Note: not writing int StaticEntity::x = 1, because redefining, not defining or redeclaring", 0)
    StaticEntity::x = 1;
    PRINT("Then, printing from e1 and e2 gives same thing", 0)
    e1.Print("e1");
    e2.Print("e2");
    PRINT("And really, it's equivalent to doing StaticEntity::Print() twice but with different arguments to Print() i.e. \"e1\" and \"e2\"", 0)
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
    PRINT("We can also manually call the destructor if we want for whatever reason with instance.~CtorDtor();", 0)
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
    PRINT("Character has two floats, so size of Character via sizeof(Character) is", sizeof(Character))
    PRINT("Enemy has one int plus whatever in Character, so size of Enemy via sizeof(Enemy) is", sizeof(Enemy))
}

void virtualFunction() {
    Enemy enemy;
    enemy.PrintPosition();
    enemy.PrintHealth();
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
}