pragma solidity ^0.4.18;
contract Charity_For_My_Friend{
    address owner;
    function Charity_For_My_Friend() {
        owner = msg.sender;
    }
    function kill() {
        require(msg.sender == owner);
        selfdestruct(owner);
    }
    function () payable {}
}