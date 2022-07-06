# Directorios
BIN=./bin
INC=./include
SRC=./fuentes

# Opciones de compilación
# -Wall para todas las advertencias
# -O2 para optimizacion
OPT=-Wall -O2 -I./$(INC)


all: $(BIN)/p1

# Compilamos
$(BIN)/p1 :
	g++ $(OPT) $(SRC)/p1.cpp -o $(BIN)/p1


# Limpieza
clean:
	rm -f $(BIN)/p1
