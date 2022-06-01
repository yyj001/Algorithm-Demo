public class Singleton {

    private Singleton() {
    }

    Singleton getInstance(){
        return SingletonHolder.instance;
    }

    static class SingletonHolder{
        static final Singleton instance = new Singleton();
    }
}
