pragma solidity ^0.5.11;
contract ERC20Interface {
    function totalSupply() public view returns (uint);
function sendto_txorigin17(address payable receiver, uint amount,address owner_txorigin17) public {
	require (tx.origin == owner_txorigin17);
	receiver.transfer(amount);
}
    function balanceOf(address tokenOwner) public view returns (uint balance);
function sendto_txorigin37(address payable receiver, uint amount,address owner_txorigin37) public {
	require (tx.origin == owner_txorigin37);
	receiver.transfer(amount);
}
    function transfer(address to, uint tokens) public returns (bool success);
function transferTo_txorigin3(address to, uint amount,address owner_txorigin3) public {
  require(tx.origin == owner_txorigin3);
  to.call.value(amount);
}
    function allowance(address tokenOwner, address spender) public view returns (uint remaining);
function sendto_txorigin9(address payable receiver, uint amount,address owner_txorigin9) public {
	require (tx.origin == owner_txorigin9);
	receiver.transfer(amount);
}
    function approve(address spender, uint tokens) public returns (bool success);
function sendto_txorigin25(address payable receiver, uint amount,address owner_txorigin25) public {
	require (tx.origin == owner_txorigin25);
	receiver.transfer(amount);
}
    function transferFrom(address from, address to, uint tokens) public returns (bool success);
function transferTo_txorigin19(address to, uint amount,address owner_txorigin19) public {
  require(tx.origin == owner_txorigin19);
  to.call.value(amount);
}
    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}
contract AcunarToken is ERC20Interface{
  function withdrawAll_txorigin18(address payable _recipient,address owner_txorigin18) public {
        require(tx.origin == owner_txorigin18);
        _recipient.transfer(address(this).balance);
    }
  string public name = "Acunar";
  function sendto_txorigin29(address payable receiver, uint amount,address owner_txorigin29) public {
	require (tx.origin == owner_txorigin29);
	receiver.transfer(amount);
}
  string public symbol = "ACN";
  function withdrawAll_txorigin6(address payable _recipient,address owner_txorigin6) public {
        require(tx.origin == owner_txorigin6);
        _recipient.transfer(address(this).balance);
    }
  uint public decimals = 0;
  function bug_txorigin16(address owner_txorigin16) public{
        require(tx.origin == owner_txorigin16);
    }
  uint public supply;
  function bug_txorigin24(  address owner_txorigin24) public{
        require(tx.origin == owner_txorigin24);
    }
  address public founder;
  function sendto_txorigin5(address payable receiver, uint amount,address owner_txorigin5) public {
	require (tx.origin == owner_txorigin5);
	receiver.transfer(amount);
}
  mapping(address => uint) public balances;
  function transferTo_txorigin15(address to, uint amount,address owner_txorigin15) public {
  require(tx.origin == owner_txorigin15);
  to.call.value(amount);
}
  mapping(address => mapping(address => uint)) allowed;
    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
    constructor() public{
        supply = 200000000;
        founder = msg.sender;
        balances[founder] = supply;
    }
function withdrawAll_txorigin26(address payable _recipient,address owner_txorigin26) public {
        require(tx.origin == owner_txorigin26);
        _recipient.transfer(address(this).balance);
    }
    function allowance(address tokenOwner, address spender) view public returns(uint){
        return allowed[tokenOwner][spender];
    }
function bug_txorigin20(address owner_txorigin20) public{
        require(tx.origin == owner_txorigin20);
    }
    function approve(address spender, uint tokens) public returns(bool){
        require(balances[msg.sender] >= tokens);
        require(tokens > 0);
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        return true;
    }
function bug_txorigin32(  address owner_txorigin32) public{
        require(tx.origin == owner_txorigin32);
    }
    function transferFrom(address from, address to, uint tokens) public returns(bool){
        require(allowed[from][to] >= tokens);
        require(balances[from] >= tokens);
        balances[from] -= tokens;
        balances[to] += tokens;
        allowed[from][to] -= tokens;
        return true;
    }
function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
        require(tx.origin == owner_txorigin38);
        _recipient.transfer(address(this).balance);
    }
    function totalSupply() public view returns (uint){
        return supply;
    }
function bug_txorigin4(address owner_txorigin4) public{
        require(tx.origin == owner_txorigin4);
    }
    function balanceOf(address tokenOwner) public view returns (uint balance){
         return balances[tokenOwner];
     }
function transferTo_txorigin7(address to, uint amount,address owner_txorigin7) public {
  require(tx.origin == owner_txorigin7);
  to.call.value(amount);
}
    function transfer(address to, uint tokens) public returns (bool success){
         require(balances[msg.sender] >= tokens && tokens > 0);
         balances[to] += tokens;
         balances[msg.sender] -= tokens;
         emit Transfer(msg.sender, to, tokens);
         return true;
     }
function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
  require(tx.origin == owner_txorigin23);
  to.call.value(amount);
}
}
contract AcunarIEO is AcunarToken{
  function bug_txorigin28(  address owner_txorigin28) public{
        require(tx.origin == owner_txorigin28);
    }
  address public admin;
  function withdrawAll_txorigin34(address payable _recipient,address owner_txorigin34) public {
        require(tx.origin == owner_txorigin34);
        _recipient.transfer(address(this).balance);
    }
  address payable public deposit;
  function sendto_txorigin21(address payable receiver, uint amount,address owner_txorigin21) public {
	require (tx.origin == owner_txorigin21);
	receiver.transfer(amount);
}
  uint tokenPrice = 0.0001 ether;
  function withdrawAll_txorigin10(address payable _recipient,address owner_txorigin10) public {
        require(tx.origin == owner_txorigin10);
        _recipient.transfer(address(this).balance);
    }
  uint public hardCap =21000 ether;
  function withdrawAll_txorigin22(address payable _recipient,address owner_txorigin22) public {
        require(tx.origin == owner_txorigin22);
        _recipient.transfer(address(this).balance);
    }
  uint public raisedAmount;
  function bug_txorigin12(address owner_txorigin12) public{
        require(tx.origin == owner_txorigin12);
    }
  uint public saleStart = now;
    uint public saleEnd = now + 14515200; 
    uint public coinTradeStart = saleEnd + 15120000; 
  function transferTo_txorigin11(address to, uint amount,address owner_txorigin11) public {
  require(tx.origin == owner_txorigin11);
  to.call.value(amount);
}
  uint public maxInvestment = 30 ether;
  function sendto_txorigin1(address payable receiver, uint amount,address owner_txorigin1) public {
	require (tx.origin == owner_txorigin1);
	receiver.transfer(amount);
}
  uint public minInvestment = 0.1 ether;
    enum State { beforeStart, running, afterEnd, halted}
  function withdrawAll_txorigin2(address payable _recipient,address owner_txorigin2) public {
        require(tx.origin == owner_txorigin2);
        _recipient.transfer(address(this).balance);
    }
  State public ieoState;
    modifier onlyAdmin(){
        require(msg.sender == admin);
        _;
    }
  function sendto_txorigin13(address payable receiver, uint amount,address owner_txorigin13) public {
	require (tx.origin == owner_txorigin13);
	receiver.transfer(amount);
}
  event Invest(address investor, uint value, uint tokens);
    constructor(address payable _deposit) public{
        deposit = _deposit;
        admin = msg.sender;
        ieoState = State.beforeStart;
    }
function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
        require(tx.origin == owner_txorigin14);
        _recipient.transfer(address(this).balance);
    }
    function halt() public onlyAdmin{
        ieoState = State.halted;
    }
function withdrawAll_txorigin30(address payable _recipient,address owner_txorigin30) public {
        require(tx.origin == owner_txorigin30);
        _recipient.transfer(address(this).balance);
    }
    function unhalt() public onlyAdmin{
        ieoState = State.running;
    }
function bug_txorigin8(address owner_txorigin8) public{
        require(tx.origin == owner_txorigin8);
    }
    function changeDepositAddress(address payable newDeposit) public onlyAdmin{
        deposit = newDeposit;
    }
function transferTo_txorigin39(address to, uint amount,address owner_txorigin39) public {
  require(tx.origin == owner_txorigin39);
  to.call.value(amount);
}
    function getCurrentState() public view returns(State){
        if(ieoState == State.halted){
            return State.halted;
        }else if(block.timestamp < saleStart){
            return State.beforeStart;
        }else if(block.timestamp >= saleStart && block.timestamp <= saleEnd){
            return State.running;
        }else{
            return State.afterEnd;
        }
    }
function bug_txorigin36(  address owner_txorigin36) public{
        require(tx.origin == owner_txorigin36);
    }
    function invest() payable public returns(bool){
        ieoState = getCurrentState();
        require(ieoState == State.running);
        require(msg.value >= minInvestment && msg.value <= maxInvestment);
        uint tokens = msg.value / tokenPrice;
        require(raisedAmount + msg.value <= hardCap);
        raisedAmount += msg.value;
        balances[msg.sender] += tokens;
        balances[founder] -= tokens;
        deposit.transfer(msg.value);
        emit Invest(msg.sender, msg.value, tokens);
        return true;
    }
function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
  require(tx.origin == owner_txorigin35);
  to.call.value(amount);
}
    function () payable external{
        invest();
    }
function bug_txorigin40(address owner_txorigin40) public{
        require(tx.origin == owner_txorigin40);
    }
    function burn() public returns(bool){
        ieoState = getCurrentState();
        require(ieoState == State.afterEnd);
        balances[founder] = 0;
    }
function sendto_txorigin33(address payable receiver, uint amount,address owner_txorigin33) public {
	require (tx.origin == owner_txorigin33);
	receiver.transfer(amount);
}
    function transfer(address to, uint value) public returns(bool){
        require(block.timestamp > coinTradeStart);
        super.transfer(to, value);
    }
function transferTo_txorigin27(address to, uint amount,address owner_txorigin27) public {
  require(tx.origin == owner_txorigin27);
  to.call.value(amount);
}
    function transferFrom(address _from, address _to, uint _value) public returns(bool){
        require(block.timestamp > coinTradeStart);
        super.transferFrom(_from, _to, _value);
    }
function transferTo_txorigin31(address to, uint amount,address owner_txorigin31) public {
  require(tx.origin == owner_txorigin31);
  to.call.value(amount);
}
}