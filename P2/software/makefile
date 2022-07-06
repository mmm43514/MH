# Directorios
BIN=./bin
INC=./include
SRC=./fuentes

# Opciones de compilación
# -Wall para todas las advertencias
# -O2 para optimizacion
OPT=-Wall -O2 -I./$(INC)


all: $(BIN)/p2

# Compilamos
$(BIN)/p2 :
	g++ $(OPT) $(SRC)/p2.cpp -o $(BIN)/p2


# Limpieza
clean:
	rm -f $(BIN)/p2
