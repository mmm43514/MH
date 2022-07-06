# Directorios
BIN=./bin
INC=./include
SRC=./fuentes

# Opciones de compilación
# -Wall para todas las advertencias
# -O2 para optimizacion
OPT=-Wall -O2 -I./$(INC)


all: $(BIN)/pwo

# Compilamos
$(BIN)/pwo:
	g++ $(OPT) $(SRC)/pwo.cpp -o $(BIN)/pwo


# Limpieza
clean:
	rm -f $(BIN)/pwo
