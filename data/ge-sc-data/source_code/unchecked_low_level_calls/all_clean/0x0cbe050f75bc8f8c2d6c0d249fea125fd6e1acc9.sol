pragma solidity ^0.4.25;
contract Caller {
    function callAddress(address a) {
        a.call();
    }
}